"""
Notion Correction Import Service - Reads user corrections from Notion and applies them to the graph.

Runs a multi-step pipeline:
  1. Discover - Find child databases under the parent Notion page
  2. Scan     - Query each database for rows with "Needs Review" checked
  3. Apply    - For each flagged row:
                a) Call graphiti.add_episode() to correct the knowledge graph
                b) Use a LangGraph + Notion MCP agent to update or delete
                   the Notion row based on the corrected data

The pipeline runs as a background task; callers poll a job store for progress.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EpisodeType
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from notion_client import AsyncClient as NotionAsyncClient
from notion_client.client import ClientOptions
from opentelemetry import trace

from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.observability import (
    anonymize_id,
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)
from app.services.notion_correction_job_store import (
    complete_notion_correction_job,
    fail_notion_correction_job,
    update_notion_correction_step,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Notion API rate-limit delay (3 req/s limit → ~0.35s between calls)
_NOTION_RATE_LIMIT_DELAY = 0.35

# ---------------------------------------------------------------------------
# Prompts & templates
# ---------------------------------------------------------------------------

_CORRECTION_EXTRACTION_TEMPLATE = (
    "This episode is a user-initiated correction to previously recorded knowledge. "
    "Extract ONLY the corrected facts as new edges. When you encounter statements "
    "about what was 'previously' or 'incorrectly' recorded, do NOT extract those "
    "as new facts — they represent the old state that should be contradicted. "
    "Focus on extracting the corrected/current state as declarative facts. "
    "IMPORTANT: All extracted facts, entity names, and summaries MUST be written in {language}."
)

_ROW_UPDATE_PROMPT = """\
You are a Notion database editor processing a user correction.

## Language
All property values MUST be in **{language}**.

## Page to update
- **Page ID**: `{page_id}`
- **Database**: {category_name}

## Current row properties
{current_props}

## Column schema (property name → type)
{prop_schema}

## Correction from knowledge graph

User correction: {correction_notes}

Updated node summaries:
{node_summaries}

New/updated facts:
{new_facts}

Invalidated facts (no longer true):
{invalidated_facts}

## Instructions

Based on the correction data above, choose ONE action:

**Option A — UPDATE the row** (if the entity is still relevant):
Use `API-patch-page` with page_id `{page_id}` to update the row.
- Update ALL properties that are affected by the correction data.
- Set "Needs Review" to false (checkbox).
- Clear "Correction Notes" (set to empty rich_text).
- Keep properties unchanged if the correction doesn't affect them.

**Option B — DELETE the row** (if the entity is no longer relevant or valid):
If the correction means this entity should be removed (e.g., user said it's \
irrelevant, or all facts about it were invalidated), use `API-delete-block` \
with block_id `{page_id}` to archive the row.

Format property values per Notion API:
- title → {{"title": [{{"text": {{"content": "value"}}}}]}}
- rich_text → {{"rich_text": [{{"text": {{"content": "value"}}}}]}}
- select → {{"select": {{"name": "value"}}}}
- number → {{"number": 123}}
- date → {{"date": {{"start": "YYYY-MM-DD"}}}}
- checkbox → {{"checkbox": false}}
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CorrectionItem:
    """A single row flagged for correction in a Notion database."""

    category_name: str
    database_id: str
    page_id: str
    title: str
    properties: dict[str, str]
    property_types: dict[str, str]
    correction_notes: str


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _extract_property_value(prop: dict[str, Any]) -> str:
    """Extract plain text value from a Notion property object."""
    prop_type = prop.get("type", "")
    if prop_type == "title":
        return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    elif prop_type == "rich_text":
        return "".join(t.get("plain_text", "") for t in prop.get("rich_text", []))
    elif prop_type == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    elif prop_type == "number":
        num = prop.get("number")
        return str(num) if num is not None else ""
    elif prop_type == "date":
        d = prop.get("date")
        return d.get("start", "") if d else ""
    elif prop_type == "checkbox":
        return str(prop.get("checkbox", False))
    return ""


def _build_episode_body(item: CorrectionItem) -> str:
    """Build a natural language episode body for Graphiti's add_episode."""
    props_text = ", ".join(
        f'{k}: "{v}"'
        for k, v in item.properties.items()
        if k not in ("Needs Review", "Correction Notes") and v
    )

    return (
        f'CORRECTION for [{item.category_name}], entity "{item.title}":\n'
        f"Previously recorded properties: {props_text}\n"
        f"User correction: {item.correction_notes}\n"
        f"Apply this correction to the knowledge graph."
    )


def _build_row_update_prompt(
    item: CorrectionItem,
    episode_result: AddEpisodeResults,
    language: str,
) -> str:
    """Build the prompt for the MCP agent to update/delete the Notion row."""
    current_props = "\n".join(
        f"- {k} ({item.property_types.get(k, 'unknown')}): {v}"
        for k, v in item.properties.items()
        if k not in ("Needs Review", "Correction Notes")
    )

    prop_schema = "\n".join(
        f"- {k}: {t}"
        for k, t in item.property_types.items()
        if k not in ("Needs Review", "Correction Notes")
    )

    node_summaries = "\n".join(
        f"- {node.name}: {node.summary}"
        for node in episode_result.nodes
        if node.summary
    ) or "(none)"

    new_facts = "\n".join(
        f"- {e.fact}"
        for e in episode_result.edges
        if e.fact and not e.invalid_at
    ) or "(none)"

    invalidated_facts = "\n".join(
        f"- {e.fact}"
        for e in episode_result.edges
        if e.fact and e.invalid_at
    ) or "(none)"

    return _ROW_UPDATE_PROMPT.format(
        language=language,
        page_id=item.page_id,
        category_name=item.category_name,
        current_props=current_props,
        prop_schema=prop_schema,
        correction_notes=item.correction_notes,
        node_summaries=node_summaries,
        new_facts=new_facts,
        invalidated_facts=invalidated_facts,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class NotionCorrectionService:
    """Orchestrates the Notion correction import pipeline."""

    def __init__(
        self,
        graphiti: Graphiti,
        google_api_key: str,
        max_concurrent_imports: int = 3,
    ):
        self._graphiti = graphiti
        self._google_api_key = google_api_key
        self._semaphore = asyncio.Semaphore(max_concurrent_imports)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_import(
        self,
        job_id: str,
        group_id: str,
        notion_token: str,
        page_id: str,
        page_name: str,
        language: str = "English",
    ) -> None:
        """Launch the correction import pipeline as a background task."""
        asyncio.create_task(
            self._run_pipeline(
                job_id=job_id,
                group_id=group_id,
                notion_token=notion_token,
                page_id=page_id,
                language=language,
            )
        )

    # ------------------------------------------------------------------
    # Pipeline orchestrator
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        job_id: str,
        group_id: str,
        notion_token: str,
        page_id: str,
        language: str = "English",
    ) -> None:
        async with self._semaphore:
            with tracer.start_as_current_span(
                "notion_correction.run_pipeline",
                attributes={
                    "correction.job_id": job_id,
                    "correction.group_id": anonymize_id(group_id),
                    "correction.page_id": page_id,
                },
            ) as span:
                pipeline_start = time.monotonic()
                try:
                    notion = NotionAsyncClient(
                        options=ClientOptions(
                            auth=notion_token,
                            notion_version="2022-06-28",
                        )
                    )

                    database_ids = await self._discover_databases(notion, page_id)
                    if not database_ids:
                        fail_notion_correction_job(
                            job_id,
                            error="No databases found under the specified Notion page.",
                            code="NO_DATABASES",
                        )
                        set_span_attributes(span, {"correction.no_databases": True})
                        mark_span_success(span, status="no_databases")
                        return

                    set_span_attributes(
                        span,
                        {"correction.databases_count": len(database_ids)},
                    )

                    correction_items = await self._step_scan(
                        job_id, notion, database_ids,
                    )

                    if not correction_items:
                        complete_notion_correction_job(
                            job_id,
                            corrections_found=0,
                            corrections_applied=0,
                            corrections_failed=0,
                            failed_corrections=None,
                            duration_ms=round(
                                (time.monotonic() - pipeline_start) * 1000, 2,
                            ),
                        )
                        set_span_attributes(span, {"correction.no_corrections": True})
                        mark_span_success(span, status="no_corrections")
                        return

                    applied, failed, failed_list = await self._step_apply(
                        job_id, notion, notion_token, correction_items,
                        group_id, language,
                    )

                    duration_ms = round(
                        (time.monotonic() - pipeline_start) * 1000, 2,
                    )
                    complete_notion_correction_job(
                        job_id,
                        corrections_found=len(correction_items),
                        corrections_applied=applied,
                        corrections_failed=failed,
                        failed_corrections=failed_list if failed_list else None,
                        duration_ms=duration_ms,
                    )
                    set_span_attributes(
                        span,
                        {
                            "correction.found": len(correction_items),
                            "correction.applied": applied,
                            "correction.failed": failed,
                            "duration_ms": duration_ms,
                        },
                    )
                    mark_span_success(span)

                except Exception as exc:
                    category, code = classify_error(exc)
                    fail_notion_correction_job(
                        job_id,
                        error=str(exc)[:500],
                        code=code,
                    )
                    mark_span_error(span, exc, category=category, code=code)
                    logger.exception(
                        "Notion correction pipeline failed for job %s", job_id,
                    )

    # ------------------------------------------------------------------
    # Step 0: Discover child databases under the parent page
    # ------------------------------------------------------------------

    async def _discover_databases(
        self,
        notion: NotionAsyncClient,
        page_id: str,
    ) -> dict[str, str]:
        """List all child databases under the parent page.

        Returns a dict mapping database title to database ID.
        """
        with tracer.start_as_current_span("notion_correction.discover_databases") as span:
            start = time.monotonic()
            database_ids: dict[str, str] = {}
            has_more = True
            start_cursor: str | None = None

            while has_more:
                response = await notion.blocks.children.list(
                    block_id=page_id,
                    start_cursor=start_cursor,
                )
                for block in response.get("results", []):
                    if block.get("type") == "child_database":
                        title = block.get("child_database", {}).get("title", "")
                        if title:
                            database_ids[title] = block["id"]

                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
                if has_more:
                    await asyncio.sleep(_NOTION_RATE_LIMIT_DELAY)

            set_span_attributes(
                span,
                {
                    "correction.databases_discovered": len(database_ids),
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            logger.info(
                "Discovered %d databases under page %s: %s",
                len(database_ids),
                page_id,
                list(database_ids.keys()),
            )
            return database_ids

    # ------------------------------------------------------------------
    # Step 1: Scan Notion databases for flagged rows
    # ------------------------------------------------------------------

    async def _step_scan(
        self,
        job_id: str,
        notion: NotionAsyncClient,
        database_ids: dict[str, str],
    ) -> list[CorrectionItem]:
        update_notion_correction_step(job_id, "scanning")

        with tracer.start_as_current_span("notion_correction.scan") as span:
            start = time.monotonic()
            all_items: list[CorrectionItem] = []

            for category_name, db_id in database_ids.items():
                items = await self._query_flagged_rows(
                    notion, db_id, category_name,
                )
                all_items.extend(items)
                await asyncio.sleep(_NOTION_RATE_LIMIT_DELAY)

            update_notion_correction_step(
                job_id,
                "scanning",
                databases_scanned=len(database_ids),
                corrections_found=len(all_items),
            )
            set_span_attributes(
                span,
                {
                    "correction.databases_scanned": len(database_ids),
                    "correction.corrections_found": len(all_items),
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return all_items

    async def _query_flagged_rows(
        self,
        notion: NotionAsyncClient,
        db_id: str,
        category_name: str,
    ) -> list[CorrectionItem]:
        """Query a single Notion database for rows with Needs Review checked."""
        items: list[CorrectionItem] = []
        has_more = True
        start_cursor: str | None = None

        while has_more:
            body: dict[str, Any] = {
                "filter": {
                    "property": "Needs Review",
                    "checkbox": {"equals": True},
                },
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            response = await notion.request(
                path=f"databases/{db_id}/query",
                method="POST",
                body=body,
            )
            for page in response.get("results", []):
                item = self._parse_page(page, category_name, db_id)
                if item is not None:
                    items.append(item)

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")
            if has_more:
                await asyncio.sleep(_NOTION_RATE_LIMIT_DELAY)

        return items

    def _parse_page(
        self,
        page: dict[str, Any],
        category_name: str,
        db_id: str,
    ) -> CorrectionItem | None:
        """Parse a Notion page into a CorrectionItem, or None if unusable."""
        properties = page.get("properties", {})
        page_id = page.get("id", "")

        prop_values: dict[str, str] = {}
        prop_types: dict[str, str] = {}
        title_value = ""
        correction_notes = ""

        for prop_name, prop_data in properties.items():
            value = _extract_property_value(prop_data)
            prop_type = prop_data.get("type", "unknown")

            if prop_name == "Correction Notes":
                correction_notes = value.strip()
                continue
            if prop_name == "Needs Review":
                continue

            prop_values[prop_name] = value
            prop_types[prop_name] = prop_type
            if prop_type == "title":
                title_value = value

        if not correction_notes:
            logger.warning(
                "Skipping row %s in %s: flagged but no Correction Notes",
                page_id,
                category_name,
            )
            return None

        return CorrectionItem(
            category_name=category_name,
            database_id=db_id,
            page_id=page_id,
            title=title_value or "(untitled)",
            properties=prop_values,
            property_types=prop_types,
            correction_notes=correction_notes,
        )

    # ------------------------------------------------------------------
    # Step 2: Apply corrections to graph + update Notion rows
    # ------------------------------------------------------------------

    async def _step_apply(
        self,
        job_id: str,
        notion: NotionAsyncClient,
        notion_token: str,
        items: list[CorrectionItem],
        group_id: str,
        language: str = "English",
    ) -> tuple[int, int, list[dict]]:
        update_notion_correction_step(
            job_id, "applying", corrections_found=len(items),
        )

        with tracer.start_as_current_span("notion_correction.apply") as span:
            start = time.monotonic()
            applied = 0
            failed = 0
            failed_list: list[dict] = []

            async with self._create_notion_agent(notion_token) as agent:
                for item in items:
                    try:
                        episode_result = await self._correct_graph(
                            item, group_id, language,
                        )
                        await self._update_notion_row(
                            agent, item, episode_result, language,
                        )
                        await self._reset_review_fields(notion, item.page_id)
                        applied += 1
                    except Exception as exc:
                        failed += 1
                        failed_list.append(
                            {
                                "category": item.category_name,
                                "title": item.title,
                                "error": str(exc)[:200],
                            }
                        )
                        logger.warning(
                            "Correction failed for %s/%s: %s",
                            item.category_name,
                            item.title,
                            exc,
                        )

                    update_notion_correction_step(
                        job_id,
                        "applying",
                        corrections_applied=applied,
                        corrections_failed=failed,
                    )

            set_span_attributes(
                span,
                {
                    "correction.applied": applied,
                    "correction.failed": failed,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return applied, failed, failed_list

    # ------------------------------------------------------------------
    # MCP agent lifecycle
    # ------------------------------------------------------------------

    def _create_notion_agent(self, notion_token: str) -> _NotionAgentContext:
        """Create a context manager that starts the Notion MCP server and returns a LangGraph agent.

        Usage:
            async with self._create_notion_agent(token) as agent:
                await agent.astream(...)
        """
        return _NotionAgentContext(
            google_api_key=self._google_api_key,
            notion_token=notion_token,
        )

    # ------------------------------------------------------------------
    # Graph correction
    # ------------------------------------------------------------------

    async def _correct_graph(
        self,
        item: CorrectionItem,
        group_id: str,
        language: str,
    ) -> AddEpisodeResults:
        """Apply a single correction to the knowledge graph via Graphiti."""
        episode_body = _build_episode_body(item)
        extraction_instructions = _CORRECTION_EXTRACTION_TEMPLATE.format(
            language=language,
        )

        with tracer.start_as_current_span(
            "notion_correction.correct_graph",
            attributes={
                "correction.category": item.category_name,
                "correction.title": item.title,
                "correction.text_length_chars": len(episode_body),
            },
        ) as span:
            try:
                logger.info(
                    "Applying graph correction for %s/%s (page %s)",
                    item.category_name,
                    item.title,
                    item.page_id,
                )
                result = await self._graphiti.add_episode(
                    name="notion_correction",
                    episode_body=episode_body,
                    source=EpisodeType.text,
                    source_description=(
                        f"User correction from Notion review ({item.category_name})"
                    ),
                    group_id=group_id,
                    reference_time=datetime.now(),
                    custom_extraction_instructions=extraction_instructions,
                )
                set_span_attributes(
                    span,
                    {
                        "correction.nodes_count": len(result.nodes),
                        "correction.edges_count": len(result.edges),
                    },
                )
                mark_span_success(span)
                return result
            except Exception as exc:
                category, code = classify_error(exc)
                mark_span_error(span, exc, category=category, code=code)
                raise

    # ------------------------------------------------------------------
    # Notion row update via MCP agent
    # ------------------------------------------------------------------

    async def _update_notion_row(
        self,
        agent: Any,
        item: CorrectionItem,
        episode_result: AddEpisodeResults,
        language: str,
    ) -> None:
        """Use the MCP agent to intelligently update or delete the Notion row."""
        prompt = _build_row_update_prompt(item, episode_result, language)

        with tracer.start_as_current_span(
            "notion_correction.update_notion_row",
            attributes={
                "correction.category": item.category_name,
                "correction.title": item.title,
                "correction.page_id": item.page_id,
            },
        ) as span:
            try:
                start = time.monotonic()
                async for _step in agent.astream(
                    {"messages": [("user", prompt)]}
                ):
                    pass
                set_span_attributes(
                    span,
                    {"duration_ms": round((time.monotonic() - start) * 1000, 2)},
                )
                mark_span_success(span)
            except Exception as exc:
                category, code = classify_error(exc)
                mark_span_error(span, exc, category=category, code=code)
                raise

    async def _reset_review_fields(
        self,
        notion: NotionAsyncClient,
        page_id: str,
    ) -> None:
        """Reset 'Needs Review' and 'Correction Notes' via the Notion SDK.

        This guarantees the fields are cleared regardless of whether the MCP
        agent included the reset in its update call.
        """
        await notion.request(
            path=f"pages/{page_id}",
            method="PATCH",
            body={
                "properties": {
                    "Needs Review": {"checkbox": False},
                    "Correction Notes": {
                        "rich_text": [],
                    },
                },
            },
        )


# ---------------------------------------------------------------------------
# MCP agent context manager
# ---------------------------------------------------------------------------


class _NotionAgentContext:
    """Async context manager that manages the Notion MCP server lifecycle."""

    def __init__(self, google_api_key: str, notion_token: str):
        self._google_api_key = google_api_key
        self._notion_token = notion_token
        self._stdio_cm: Any = None
        self._session_cm: Any = None

    async def __aenter__(self) -> Any:
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={**os.environ, "NOTION_TOKEN": self._notion_token},
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        tools = await load_mcp_tools(session)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=self._google_api_key,
        )
        return create_react_agent(llm, tools)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)
