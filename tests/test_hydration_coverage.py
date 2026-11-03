import random
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import COMPILER_VERSION, HydrationCompiler, fact_hash
from app.services.hydration_consolidation import apply_plan
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.hydration_selection import (
    CORRECTION_FACTS,
    DIVERSITY_QUERY,
    HEAD_QUERY,
    HUB_QUERY,
    INPUT_CHARS,
    LEGACY_QUERY,
    MAX_FACTS,
    RECENT_FACTS,
    RECENT_QUERY,
    STATS_QUERY,
    VALIDATE_QUERY,
)
from app.services.hydration_store import HydrationStore


def edge(cid, **extra):
    return {
        "uuid": cid,
        "fact": f"Detalle íntegro del recuerdo {cid}.",
        "source_uuid": "user",
        "target_uuid": cid,
        "valid_at": None,
        "invalid_at": None,
        "date": "2026-11-02",
        "date_kind": "episodio",
        "priority": 1,
        "correction": False,
        "episode_ids": ["ep"],
        **extra,
    }


def test_large_input_is_capped_before_model_and_keeps_whole_priority_facts():
    edges = [edge(f"f{i:04}", fact=f"Dato {i} " + "x" * 400, priority=2) for i in range(5000)]
    edges += [edge("correction", correction=True, fact="Julia es mi prima, no mi hija.")]
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(edges)
    assert len(candidates) <= MAX_FACTS
    assert sum((len(c["quote"]) for c in candidates)) <= INPUT_CHARS
    assert candidates[0]["quote"] == "Julia es mi prima, no mi hija."
    originals = {e["uuid"]: e["fact"] for e in edges}
    assert all((c["quote"] == originals[c["id"]] for c in candidates))
    rendered, _ = compiler.render(apply_plan(candidates, {}), edges, GraphStats(1000, 5000))
    assert len(rendered.compilation_text) <= compiler.char_budget
    assert set(rendered.metadata.included_edge_ids) <= {c["id"] for c in candidates}
    assert "correction" in rendered.metadata.included_edge_ids
    assert "f4999" not in rendered.metadata.included_edge_ids


def test_fingerprint_ignores_provenance_and_ranking_but_tracks_semantic_changes():
    compiler = HydrationCompiler(None)
    rows = [edge("b"), edge("a")]
    first = compiler.fingerprint(compiler.fact_candidates(rows))
    shuffled = [{**e, "date": "2026-11-03", "episode_ids": ["other"], "priority": 2} for e in rows]
    random.Random(42).shuffle(shuffled)
    assert first == compiler.fingerprint(compiler.fact_candidates(shuffled))
    for change in (
        {"fact": "Dato corregido"},
        {"source_uuid": "cousin"},
        {"valid_at": "2026-11-01"},
        {"correction": True},
    ):
        changed = [{**rows[0], **change}, rows[1]]
        assert first != compiler.fingerprint(compiler.fact_candidates(changed))


def test_exact_duplicates_have_stable_primary_and_cover_only_matching_subject_and_time():
    compiler = HydrationCompiler(None)
    rows = [
        edge("z", fact="Ana tiene un gato.", target_uuid="cat"),
        edge("a", fact="Ana tiene un gato.", target_uuid="cat", date="2026-11-01"),
        edge("cousin", fact="Ana tiene un gato.", target_uuid="cat", source_uuid="other"),
        edge("past", fact="Ana tiene un gato.", target_uuid="cat", valid_at="2025-01-01"),
    ]
    first = compiler.fact_candidates(rows)
    assert first == compiler.fact_candidates(list(reversed(rows)))
    assert next((c for c in first if c["id"] == "a"))["duplicate_ids"] == ["z"]
    assert len(first) == 3


def test_programmatic_budget_prioritizes_corrections_globally_and_uses_whole_lines():
    compiler = HydrationCompiler(None, char_budget=1200)
    rows = [edge(f"old{i}", priority=2, date="2015-01-01") for i in range(30)]
    rows += [edge("correction", correction=True, fact="Mi prima Julia estudia ingeniería.")]
    candidates = compiler.fact_candidates(rows)
    plan = {"sections": {c["id"]: f"Tema {i}" for i, c in enumerate(candidates)}}
    result, _ = compiler.render(apply_plan(candidates, plan), rows, GraphStats(100, 100))
    assert result.metadata.included_edge_ids[0] == "correction"
    assert len(result.compilation_text) <= 1200
    assert "Mi prima Julia estudia ingeniería." in result.compilation_text
    assert len(result.metadata.included_edge_ids) < len(rows)
    assert result.metadata.included_node_ids == []


async def test_selection_uses_bounded_queries_isolates_user_and_fills_legacy_gaps():
    calls = []

    async def rows(query, **params):
        calls.append((query, params))
        assert params["user"] == "owner"
        if query == HEAD_QUERY:
            return [{"uuid": "ep"}]
        if query == RECENT_QUERY:
            return (
                [edge("correction", correction=True)] if params["corrections"] else [edge("recent")]
            )
        if query == LEGACY_QUERY:
            assert set(params["excluded"]) == {"correction", "recent"}
            assert params["limit"] == RECENT_FACTS - 1
            return [edge("legacy", episode_ids=[])]
        if query == HUB_QUERY:
            assert params["hub_candidates"] == 30 and params["per_hub"] == 8
            return [{"facts": [edge("hub", episode_ids=[]), edge("recent")]}]
        if query == DIVERSITY_QUERY:
            assert params["hubs"] == 2 and params["per_hub"] == 400
            return [{"facts": [edge("diverse", episode_ids=[])]}]
        assert query == STATS_QUERY
        return [{"entities": 1000, "relationships": 5000}]

    store = HydrationStore(None)
    store.rows = rows
    selected, head, stats = await store.select("owner")
    assert {e["uuid"] for e in selected} == {"correction", "recent", "legacy", "hub", "diverse"}
    assert head == "ep" and stats.relationship_count == 5000
    assert next((e for e in selected if e["uuid"] == "hub"))["priority"] == 2
    assert len(calls) == 7
    for query, _params in calls:
        assert ".content" not in query and ".summary" not in query
        if query not in (STATS_QUERY,):
            assert "LIMIT" in query
    assert "r.expired_at IS NULL" in RECENT_QUERY
    assert "r.invalid_at IS NULL" in RECENT_QUERY
    assert "r:RELATES_TO {uuid:id}" in RECENT_QUERY
    assert "COUNT{(n)-[:RELATES_TO]-()}" in HUB_QUERY


async def test_full_recent_quota_skips_legacy_fill():
    store = HydrationStore(None)
    store.head = AsyncMock(return_value="ep")
    store.rows = AsyncMock(
        side_effect=[
            [],
            [edge(str(i)) for i in range(RECENT_FACTS)],
            [],
            [],
            [{"entities": 1000, "relationships": 5000}],
        ]
    )
    await store.select("owner")
    assert all((call.args[0] != LEGACY_QUERY for call in store.rows.await_args_list))
    assert store.rows.await_args_list[0].kwargs["fact_limit"] == CORRECTION_FACTS


@pytest.mark.parametrize("change", ["deleted_alias", "changed_subject", "changed_detail"])
async def test_all_covered_dependencies_revoke_when_any_original_changes(change):
    original = edge("a")
    alias = edge("alias", fact=original["fact"], target_uuid="a")
    deps = {e["uuid"]: fact_hash(e) for e in [original, alias]}
    result = HydrationResult(
        "Memory", CompilationMetadata(True, 10, compilation_revision="rev"), GraphStats(2, 2)
    )
    payload = {
        "compiler": COMPILER_VERSION,
        "deps": deps,
        "head_episode": "ep",
        "result": asdict(result),
    }
    store = HydrationStore(None)
    store.current = AsyncMock(return_value=payload)
    store.rows = AsyncMock(return_value=[original, alias])
    store.head = AsyncMock(return_value="ep")
    store.discard = AsyncMock()
    store.request = AsyncMock()
    assert await store.read("owner")
    changed = (
        [original]
        if change == "deleted_alias"
        else [
            original,
            {
                **alias,
                **(
                    {"source_uuid": "other"}
                    if change == "changed_subject"
                    else {"fact": "Other detail"}
                ),
            },
        ]
    )
    store.rows.return_value = changed
    assert await store.read("owner") is None
    store.discard.assert_awaited_once_with("owner", "rev")
    assert store.rows.await_args.args[0] == VALIDATE_QUERY


async def test_new_episode_queues_additive_rebuild_without_revoking_or_incrementing_generation():
    result = HydrationResult(
        "Memory", CompilationMetadata(True, 10, compilation_revision="rev"), GraphStats(1, 1)
    )
    store = HydrationStore(None)
    store.current = AsyncMock(
        return_value={
            "compiler": COMPILER_VERSION,
            "deps": {},
            "head_episode": "old",
            "result": asdict(result),
        }
    )
    store.validate = AsyncMock(return_value=True)
    store.head = AsyncMock(return_value="new")
    store.request = AsyncMock()
    store.discard = AsyncMock()
    assert (await store.read("owner")).compilation_text == "Memory"
    store.request.assert_awaited_once_with("owner")
    store.discard.assert_not_awaited()


def test_retired_facts_stay_out_and_fact_text_does_not_trigger_business_exclusions():
    rows = [
        edge("retired", active=False, fact="La usuaria toma X."),
        edge("expired", expired_at="2026-01-01", fact="Dato vencido."),
        edge("correction", correction=True, fact="Es mi prima quien toma X."),
        edge("literal", fact="No guardes rencor, decía mi abuela."),
    ]
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(rows)
    assert {c["id"] for c in candidates} == {"correction", "literal"}
    result, _ = compiler.render(apply_plan(candidates, {}), rows, GraphStats(3, 4))
    assert "Es mi prima" in result.compilation_text
    assert "La usuaria toma X" not in result.compilation_text
    assert "No guardes rencor" in result.compilation_text
