"""
Notion Export Service - Exports a user's knowledge graph into Notion.

Runs a multi-step pipeline:
  1. Hydrate   - Build the graph compilation via HydrationService (v1)
  2. Analyze   - Gemini designs database schemas then extracts entries
  3. Create    - Notion SDK creates one database per category
  4. Populate  - MCP agent fills databases with extracted rows (batched)
  5. Summarize - MCP agent creates a "Knowledge Graph Overview" page

Each step is wrapped in an OpenTelemetry span for Axiom observability.
The pipeline runs as a background task; callers poll a job store for progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from notion_client import AsyncClient as NotionAsyncClient
from notion_client.client import ClientOptions
from opentelemetry import trace
from pydantic import BaseModel, Field

from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.config import Settings, create_langchain_llm
from app.core.observability import (
    anonymize_id,
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)
from app.core.posthog import capture_generation, capture_span, capture_trace, new_trace_id
from app.services.hydration import HydrationService
from app.services.notion_export_job_store import (
    complete_notion_export_job,
    fail_notion_export_job,
    update_notion_export_step,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Gemini schema compatibility patch
#
# The Notion MCP server exposes tool schemas with nullable properties that
# google-genai's Schema.model_validate rejects.  This patch replaces None
# property values with a safe placeholder so tool calls don't crash.
# ---------------------------------------------------------------------------
from google.genai import types as _genai_types  # noqa: E402

_orig_schema_validate = _genai_types.Schema.model_validate.__func__
_PLACEHOLDER_SCHEMA = None


def _safe_schema_validate(cls, obj, *args, **kwargs):  # type: ignore[no-untyped-def]
    global _PLACEHOLDER_SCHEMA  # noqa: PLW0603
    if isinstance(obj, dict) and isinstance(obj.get("properties"), dict):
        props = obj["properties"]
        if any(v is None for v in props.values()):
            if _PLACEHOLDER_SCHEMA is None:
                _PLACEHOLDER_SCHEMA = _orig_schema_validate(
                    cls,
                    {"type": "STRING", "description": "JSON-encoded value"},
                )
            obj = {
                **obj,
                "properties": {
                    k: (_PLACEHOLDER_SCHEMA if v is None else v)
                    for k, v in props.items()
                },
            }
    return _orig_schema_validate(cls, obj, *args, **kwargs)


_genai_types.Schema.model_validate = classmethod(_safe_schema_validate)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Structured output models for Gemini
# ---------------------------------------------------------------------------
class PropertyDef(BaseModel):
    """Column definition for a Notion database."""

    name: str = Field(description="Column name, e.g. 'Name', 'Role', 'Status'")
    type: str = Field(description="One of: title, rich_text, select, number, date")


class SchemaCategory(BaseModel):
    """A thematic category with its column schema (no entries yet)."""

    name: str = Field(description="Category name, e.g. 'People', 'Medications'")
    description: str = Field(description="What this category captures")
    properties: list[PropertyDef] = Field(
        description="Column definitions; exactly one must have type 'title'"
    )


class SchemaResult(BaseModel):
    """Phase 1 output: category schemas + overview (no row data)."""

    categories: list[SchemaCategory] = Field(description="3-8 thematic categories")
    overview: str = Field(description="Summary of the knowledge graph")


class EntryData(BaseModel):
    """A single row to insert into a Notion database."""

    values: dict[str, str] = Field(description="Map of property name to string value")


class ExtractionResult(BaseModel):
    """Phase 2 output: all entries for a single category."""

    entries: list[EntryData] = Field(
        description="Every relevant row extracted from the knowledge graph"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SCHEMA_PROMPT = """\
You are a knowledge graph analyst designing Notion database schemas.

## Language

IMPORTANT: All output text — category names, property names, descriptions, \
and the overview — MUST be written in **{language}**.  The input may mix \
languages; translate and unify everything into {language}.

## Goal

Analyze the knowledge graph below and design between 3 and 10 Notion database \
categories that can group the concepts and life dimensions of the user data. \
For each category, define the column schema that **best captures \
the nature of the data** so the database is as insightful and useful as \
possible when browsing and filtering in Notion.

## Schema design guidelines

- Think about what Status, Type, Classification, or Role columns would make \
each category most useful.  For example, a medications category benefits from \
a "Status" (select) column with values like "Active" / "Suspended".
- Each category MUST have exactly one property with type "title" as the \
primary name/identifier.
- Available property types: title, rich_text, select, number, date.
- Prefer **select** for columns with a small set of discrete values (status, \
type, role).  Use rich_text for free-form descriptions.
- Design 3-7 properties per category (not counting the feedback columns that \
will be added automatically).
- DO NOT include entries/rows — only the category names, descriptions, and \
column schemas.

## Structural rules

1. EXCLUDE "User" and "Assistant" entities — they are system-level.
2. Section 1 ("CONCEPTUAL DEFINITIONS & IDENTITY") contains entity \
definitions — use these for most categories.
3. Section 2 ("RELATIONAL DYNAMICS & CAUSALITY") contains relationships — \
use this context to enrich definitions or detected events and connections between entities \
do not create a dedicated relationships
4. The overview should summarize the knowledge graph comprehensively.

<knowledge_graph_compilation>
{compilation_text}
</knowledge_graph_compilation>"""

EXTRACTION_PROMPT = """\
You are a knowledge graph data extractor.

## Language

IMPORTANT: All extracted values MUST be written in **{language}**.  The input \
may mix languages; translate and unify everything into {language}.

## Task

Extract **every** entity/row from the knowledge graph that belongs to the \
category described below.  Use the exact column names provided.

## Category

- **Name**: {category_name}
- **Description**: {category_description}
- **Columns**: {properties}
- **Title column**: {title_property}

## Rules

1. Every entry MUST include a value for the title column "{title_property}".
2. Every entry MUST include values for ALL columns listed above.
3. Be thorough — include every relevant entity you can find in the graph.  \
Do NOT limit, cap, or skip entries.
4. Do NOT over-summarize.  Preserve meaningful detail in each value.  If a \
description needs several sentences to capture the context accurately, use them.
5. For "select" columns, keep values short and consistent across entries \
(e.g. always use the same term for the same concept).

<knowledge_graph_compilation>
{compilation_text}
</knowledge_graph_compilation>"""

POPULATE_PROMPT = """\
You are a Notion database writer. Create rows in the database below.

## Language
All property values MUST be in **{language}**.

## Database
- **ID**: `{db_id}`
- **Name**: {category_name}

## Properties
{prop_lines}

## Entries (batch {batch_idx}/{total_batches})
{entry_lines}

## Instructions
Use `API-post-page` to create one row per entry.
Set `parent` to `{{"database_id":"{db_id}"}}`.
Format each property value per Notion API:
- title → {{"title": [{{"text": {{"content": "value"}}}}]}}
- rich_text → {{"rich_text": [{{"text": {{"content": "value"}}}}]}}
- select → {{"select": {{"name": "value"}}}}
- number → {{"number": 123}}
- date → {{"date": {{"start": "YYYY-MM-DD"}}}}

Create ALL {batch_size} entries in this batch."""

SUMMARY_PROMPT = """\
You are a Notion organizer. Create a summary page.

## Language
Write ALL page content (title, headings, body text) in **{language}**.

## Task
1. Use `API-post-page` to create a page:
   - parent: `{{"page_id":"{page_id}"}}`
   - title property: "Knowledge Graph Overview"

2. Use `API-patch-block-children` on the new page ID to add these blocks:
   a. heading_2: "Overview"
   b. paragraph: "{overview}"
   c. heading_2: "Databases"
   d. Bulleted list items (one per database):
{db_bullets}
   e. heading_2: "Feedback"
   f. paragraph: "To flag a correction, open any database row, \
toggle 'Needs Review' and write your notes in 'Correction Notes'."
"""

BATCH_SIZE = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _notion_prop_schema(prop_type: str) -> dict:
    """Return the Notion API property schema for a given type string.

    Includes explicit ``type`` key required by notion-client v3 / API 2022-06-28+.
    """
    schemas: dict[str, dict[str, Any]] = {
        "title": {"type": "title", "title": {}},
        "rich_text": {"type": "rich_text", "rich_text": {}},
        "select": {"type": "select", "select": {}},
        "number": {"type": "number", "number": {}},
        "date": {"type": "date", "date": {}},
        "checkbox": {"type": "checkbox", "checkbox": {}},
    }
    return schemas.get(prop_type, {"type": "rich_text", "rich_text": {}})


async def resolve_notion_page_id(notion: NotionAsyncClient, page_name: str) -> str:
    """Search for a Notion page by title and return its ID.

    Prefers an exact title match; falls back to the first search result.
    Raises ValueError when no page is found at all.
    """
    response = await notion.search(
        query=page_name,
        filter={"value": "page", "property": "object"},
    )
    results = response.get("results", [])
    for page in results:
        title_parts = (
            page.get("properties", {}).get("title", {}).get("title", [])
        )
        plain_title = "".join(t.get("plain_text", "") for t in title_parts)
        if plain_title.strip().lower() == page_name.strip().lower():
            return page["id"]
    if results:
        return results[0]["id"]
    raise ValueError(
        f"No Notion page named '{page_name}' found. "
        "Make sure you shared the page with your integration."
    )


def _extract_page_id_from_agent_step(step: dict) -> str | None:
    """Best-effort extraction of a page ID from MCP agent tool results."""
    if "tools" not in step:
        return None
    for msg in step["tools"].get("messages", []):
        raw = getattr(msg, "content", "")
        if isinstance(raw, list):
            try:
                raw = json.dumps(raw)
            except (TypeError, ValueError):
                raw = str(raw)
        if not isinstance(raw, str) or '"id"' not in raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict) and "id" in data:
                return data["id"]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class NotionExportService:
    """Orchestrates the graph-to-Notion export pipeline."""

    def __init__(
        self,
        hydration_service: HydrationService,
        settings: Settings,
        max_concurrent_exports: int = 3,
    ):
        self._hydration = hydration_service
        self._settings = settings
        self._semaphore = asyncio.Semaphore(max_concurrent_exports)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_export(
        self,
        job_id: str,
        user_id: str,
        notion_token: str,
        page_name: str,
        page_id: str,
        language: str,
    ) -> None:
        """Launch the export pipeline as a background task."""
        asyncio.create_task(
            self._run_pipeline(
                job_id=job_id,
                user_id=user_id,
                notion_token=notion_token,
                page_name=page_name,
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
        user_id: str,
        notion_token: str,
        page_name: str,
        page_id: str,
        language: str,
    ) -> None:
        posthog_trace_id = new_trace_id()
        async with self._semaphore:
            with tracer.start_as_current_span(
                "notion_export.run_pipeline",
                attributes={
                    "export.job_id": job_id,
                    "export.user_id": anonymize_id(user_id),
                    "export.page_name": page_name,
                    "export.language": language,
                },
            ) as span:
                pipeline_start = time.monotonic()
                try:
                    # Step 0: Clean existing content under the page
                    notion = NotionAsyncClient(
                        options=ClientOptions(
                            auth=notion_token,
                            notion_version="2022-06-28",
                        )
                    )
                    await self._step_clean_page(notion, page_id)

                    # Step 1: Hydrate
                    compilation_text = await self._step_hydrate(job_id, user_id)
                    if not compilation_text:
                        fail_notion_export_job(
                            job_id,
                            error="Knowledge graph is empty for this user.",
                            code="EMPTY_GRAPH",
                        )
                        set_span_attributes(span, {"export.empty_graph": True})
                        mark_span_success(span, status="empty_graph")
                        return

                    # Step 2: Analyze (schema design + entry extraction)
                    analysis = await self._step_analyze(
                        job_id, compilation_text, language,
                        posthog_trace_id=posthog_trace_id,
                        user_id=user_id,
                    )

                    # Step 3: Create Notion databases
                    database_ids = await self._step_create_databases(
                        job_id, analysis, notion, page_id,
                    )

                    # Steps 4-5 require the MCP agent
                    await self._run_mcp_steps(
                        job_id=job_id,
                        notion_token=notion_token,
                        analysis=analysis,
                        database_ids=database_ids,
                        page_id=page_id,
                        language=language,
                    )

                    total_entries = sum(
                        len(cat.get("entries", []))
                        for cat in analysis["categories"]
                    )
                    duration_ms = round(
                        (time.monotonic() - pipeline_start) * 1000, 2,
                    )

                    # Build summary URL from the job store entry (set in step 5)
                    from app.services.notion_export_job_store import (
                        get_notion_export_job,
                    )

                    job_entry = get_notion_export_job(job_id)
                    summary_url = (
                        job_entry.summary_page_url if job_entry else None
                    )

                    complete_notion_export_job(
                        job_id,
                        database_ids=database_ids,
                        summary_page_url=summary_url,
                        categories_count=len(analysis["categories"]),
                        entries_count=total_entries,
                        duration_ms=duration_ms,
                    )

                    set_span_attributes(
                        span,
                        {
                            "export.categories_count": len(analysis["categories"]),
                            "export.entries_count": total_entries,
                            "export.databases_count": len(database_ids),
                            "export.duration_ms": duration_ms,
                        },
                    )
                    mark_span_success(span)
                    logger.info(
                        "Notion export completed job=%s categories=%d entries=%d duration=%.0fms",
                        job_id,
                        len(analysis["categories"]),
                        total_entries,
                        duration_ms,
                    )

                    # PostHog LLM Analytics
                    capture_trace(
                        user_id,
                        posthog_trace_id,
                        name="notion_export",
                        properties={
                            "pipeline": "notion_export",
                            "categories_count": len(analysis["categories"]),
                            "entries_count": total_entries,
                        },
                    )

                except Exception as exc:
                    category, code = classify_error(exc)
                    mark_span_error(span, exc, category=category, code=code)
                    fail_notion_export_job(
                        job_id,
                        error=str(exc)[:500],
                        code=code,
                    )
                    logger.error(
                        "Notion export failed job=%s: %s", job_id, exc,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Step 0: Clean existing content under the parent page
    # ------------------------------------------------------------------

    async def _step_clean_page(
        self,
        notion: NotionAsyncClient,
        page_id: str,
    ) -> None:
        """Remove all child blocks under the parent page for a fresh export."""
        with tracer.start_as_current_span("notion_export.clean_page") as span:
            start = time.monotonic()
            deleted = 0
            has_more = True
            start_cursor: str | None = None

            while has_more:
                response = await notion.blocks.children.list(
                    block_id=page_id,
                    start_cursor=start_cursor,
                )
                for block in response.get("results", []):
                    block_id = block.get("id")
                    if block_id:
                        await notion.request(
                            path=f"blocks/{block_id}",
                            method="DELETE",
                        )
                        deleted += 1
                        await asyncio.sleep(0.35)

                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")

            set_span_attributes(
                span,
                {
                    "export.blocks_deleted": deleted,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            if deleted:
                logger.info(
                    "Cleaned %d blocks from page %s", deleted, page_id,
                )

    # ------------------------------------------------------------------
    # Step 1: Hydrate the knowledge graph
    # ------------------------------------------------------------------

    async def _step_hydrate(self, job_id: str, user_id: str) -> str:
        update_notion_export_step(job_id, "hydrating")

        with tracer.start_as_current_span(
            "notion_export.hydrate",
            attributes={"export.user_id": anonymize_id(user_id)},
        ) as span:
            start = time.monotonic()
            result = await self._hydration.build_user_knowledge(
                user_id, version="v1",
            )
            set_span_attributes(
                span,
                {
                    "export.compilation_size_chars": len(result.compilation_text),
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return result.compilation_text

    # ------------------------------------------------------------------
    # Step 2: Analyze — schema design + per-category entry extraction
    # ------------------------------------------------------------------

    async def _step_analyze(
        self,
        job_id: str,
        compilation_text: str,
        language: str,
        posthog_trace_id: str = "",
        user_id: str = "",
    ) -> dict:
        update_notion_export_step(job_id, "analyzing")

        llm = create_langchain_llm(
            self._settings, model="gemini-2.5-flash", temperature=0.2,
        )

        # Phase 2a: Design schemas
        with tracer.start_as_current_span(
            "notion_export.analyze_schemas",
        ) as span:
            start = time.monotonic()
            schema_prompt_text = SCHEMA_PROMPT.format(
                compilation_text=compilation_text,
                language=language,
            )
            schema_llm = llm.with_structured_output(SchemaResult)
            schema: SchemaResult = await schema_llm.ainvoke(schema_prompt_text)
            analysis = schema.model_dump()
            elapsed = (time.monotonic() - start) * 1000
            set_span_attributes(
                span,
                {
                    "export.categories_count": len(analysis["categories"]),
                    "duration_ms": round(elapsed, 2),
                },
            )
            mark_span_success(span)

            # PostHog LLM Analytics
            capture_generation(
                user_id or "system",
                posthog_trace_id,
                input_messages=schema_prompt_text[:3000],
                output=str(analysis.get("overview", ""))[:2000],
                model="gemini-2.5-flash",
                latency_ms=elapsed,
                properties={"step": "analyze_schemas"},
            )

        update_notion_export_step(
            job_id,
            "extracting_entries",
            categories_count=len(analysis["categories"]),
        )

        # Phase 2b: Extract entries per category
        extract_llm = llm.with_structured_output(ExtractionResult)
        total_entries = 0

        for cat in analysis["categories"]:
            with tracer.start_as_current_span(
                "notion_export.extract_entries",
                attributes={"export.category_name": cat["name"]},
            ) as span:
                start = time.monotonic()
                prop_desc = ", ".join(
                    f"{p['name']} ({p['type']})" for p in cat["properties"]
                )
                title_prop = next(
                    (
                        p["name"]
                        for p in cat["properties"]
                        if p["type"] == "title"
                    ),
                    cat["properties"][0]["name"],
                )

                extraction: ExtractionResult = await extract_llm.ainvoke(
                    EXTRACTION_PROMPT.format(
                        compilation_text=compilation_text,
                        category_name=cat["name"],
                        category_description=cat["description"],
                        properties=prop_desc,
                        title_property=title_prop,
                        language=language,
                    )
                )
                cat["entries"] = extraction.model_dump()["entries"]
                total_entries += len(cat["entries"])

                set_span_attributes(
                    span,
                    {
                        "export.entries_count": len(cat["entries"]),
                        "duration_ms": round(
                            (time.monotonic() - start) * 1000, 2,
                        ),
                    },
                )
                mark_span_success(span)

        update_notion_export_step(
            job_id, "extracting_entries", entries_count=total_entries,
        )
        return analysis

    # ------------------------------------------------------------------
    # Step 3: Create Notion databases via SDK
    # ------------------------------------------------------------------

    async def _step_create_databases(
        self,
        job_id: str,
        analysis: dict,
        notion: NotionAsyncClient,
        page_id: str,
    ) -> dict[str, str]:
        update_notion_export_step(job_id, "creating_databases")

        with tracer.start_as_current_span(
            "notion_export.create_databases",
        ) as span:
            start = time.monotonic()
            database_ids: dict[str, str] = {}

            for category in analysis["categories"]:
                properties: dict[str, Any] = {}
                for prop in category["properties"]:
                    properties[prop["name"]] = _notion_prop_schema(prop["type"])
                # Feedback columns for future correction loop
                properties["Needs Review"] = _notion_prop_schema("checkbox")
                properties["Correction Notes"] = _notion_prop_schema("rich_text")

                # notion-client v3 databases.create() silently drops
                # "properties" via its internal pick() whitelist. Use the
                # raw request method so the full body reaches the API.
                response = await notion.request(
                    path="databases",
                    method="POST",
                    body={
                        "parent": {
                            "type": "page_id",
                            "page_id": page_id,
                        },
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": category["name"]},
                            }
                        ],
                        "properties": properties,
                    },
                )
                database_ids[category["name"]] = response["id"]

            set_span_attributes(
                span,
                {
                    "export.databases_count": len(database_ids),
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return database_ids

    # ------------------------------------------------------------------
    # Steps 4-5: MCP agent scope (populate + summarize)
    # ------------------------------------------------------------------

    async def _run_mcp_steps(
        self,
        job_id: str,
        notion_token: str,
        analysis: dict,
        database_ids: dict[str, str],
        page_id: str,
        language: str,
    ) -> None:
        """Start the MCP server subprocess, then run populate + summarize."""
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={**os.environ, "NOTION_TOKEN": notion_token},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                tools = await load_mcp_tools(mcp_session)

                llm = create_langchain_llm(
                    self._settings, model="gemini-2.5-flash", temperature=0.2,
                )
                agent = create_react_agent(llm, tools)

                await self._step_populate(
                    job_id, analysis, database_ids, agent, language,
                )
                await self._step_create_summary(
                    job_id, analysis, database_ids, page_id, agent, language,
                )

    # ------------------------------------------------------------------
    # Step 4: Populate databases via MCP agent
    # ------------------------------------------------------------------

    async def _step_populate(
        self,
        job_id: str,
        analysis: dict,
        database_ids: dict[str, str],
        agent: Any,
        language: str,
    ) -> None:
        update_notion_export_step(job_id, "populating")

        with tracer.start_as_current_span("notion_export.populate") as parent_span:
            pipeline_start = time.monotonic()

            for category in analysis["categories"]:
                cat_name = category["name"]
                db_id = database_ids.get(cat_name)
                all_entries = category.get("entries", [])
                if not db_id or not all_entries:
                    continue

                prop_lines = "\n".join(
                    f"  - **{p['name']}** ({p['type']})"
                    for p in category["properties"]
                )

                total = len(all_entries)
                batches = [
                    all_entries[i : i + BATCH_SIZE]
                    for i in range(0, total, BATCH_SIZE)
                ]

                for b_idx, batch in enumerate(batches, 1):
                    with tracer.start_as_current_span(
                        "notion_export.populate_batch",
                        attributes={
                            "export.category_name": cat_name,
                            "export.batch_index": b_idx,
                            "export.batch_size": len(batch),
                            "export.total_batches": len(batches),
                        },
                    ) as span:
                        start = time.monotonic()
                        entry_lines = "\n".join(
                            f"  {i}. "
                            + ", ".join(
                                f'{k}: "{v}"'
                                for k, v in e["values"].items()
                            )
                            for i, e in enumerate(batch, 1)
                        )

                        prompt = POPULATE_PROMPT.format(
                            language=language,
                            db_id=db_id,
                            category_name=cat_name,
                            prop_lines=prop_lines,
                            batch_idx=b_idx,
                            total_batches=len(batches),
                            entry_lines=entry_lines,
                            batch_size=len(batch),
                        )

                        try:
                            async for _step in agent.astream(
                                {"messages": [("user", prompt)]}
                            ):
                                pass
                            set_span_attributes(
                                span,
                                {
                                    "duration_ms": round(
                                        (time.monotonic() - start) * 1000, 2,
                                    ),
                                },
                            )
                            mark_span_success(span)
                        except Exception as exc:
                            category_err, code = classify_error(exc)
                            mark_span_error(
                                span, exc, category=category_err, code=code,
                            )
                            logger.warning(
                                "Populate batch failed job=%s cat=%s batch=%d: %s",
                                job_id,
                                cat_name,
                                b_idx,
                                exc,
                            )

            set_span_attributes(
                parent_span,
                {
                    "duration_ms": round(
                        (time.monotonic() - pipeline_start) * 1000, 2,
                    ),
                },
            )
            mark_span_success(parent_span)

    # ------------------------------------------------------------------
    # Step 5: Create summary page via MCP agent
    # ------------------------------------------------------------------

    async def _step_create_summary(
        self,
        job_id: str,
        analysis: dict,
        database_ids: dict[str, str],
        page_id: str,
        agent: Any,
        language: str,
    ) -> None:
        update_notion_export_step(job_id, "summarizing")

        with tracer.start_as_current_span(
            "notion_export.create_summary",
        ) as span:
            start = time.monotonic()

            db_bullets = "\n".join(
                f"   - {cat['name']}: {cat['description']} "
                f"({len(cat.get('entries', []))} entries)"
                for cat in analysis["categories"]
            )
            overview = analysis.get("overview", "")

            prompt = SUMMARY_PROMPT.format(
                language=language,
                page_id=page_id,
                overview=overview,
                db_bullets=db_bullets,
            )

            summary_page_id: str | None = None
            try:
                async for step in agent.astream(
                    {"messages": [("user", prompt)]}
                ):
                    if summary_page_id is None:
                        summary_page_id = _extract_page_id_from_agent_step(step)
            except Exception as exc:
                category_err, code = classify_error(exc)
                mark_span_error(span, exc, category=category_err, code=code)
                logger.warning(
                    "Summary page creation failed job=%s: %s", job_id, exc,
                )
                return

            summary_url: str | None = None
            if summary_page_id:
                clean_id = summary_page_id.replace("-", "")
                # Strip type prefixes returned by the MCP server (e.g. "lc_")
                if "_" in clean_id:
                    clean_id = clean_id.split("_")[-1]
                summary_url = f"https://www.notion.so/{clean_id}"

            # Store the URL so the pipeline orchestrator can include it
            from app.services.notion_export_job_store import get_notion_export_job

            job_entry = get_notion_export_job(job_id)
            if job_entry:
                job_entry.summary_page_url = summary_url

            set_span_attributes(
                span,
                {
                    "export.summary_page_id": summary_page_id or "",
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
