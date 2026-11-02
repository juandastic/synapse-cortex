import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import (
    COMPILER_VERSION,
    HydrationCompiler,
    WorkPending,
    digest,
    fact_hash,
)
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.hydration_store import GroupCache, HydrationStore
from app.services.persisted_hydration import PersistedHydration


def archive(count=240):
    episodes = [
        dict(
            uuid="ep",
            source="message",
            valid_at="2026-11-02",
            content="User: Mi historia personal completa.",
        )
    ]
    facts = [
        dict(
            uuid=f"fact-{i:04}",
            fact=f"Detalle personal {i}: fecha 2026-11-02, importe {1000 + i} y persona Julia.",
            episodes=["ep"],
            source_uuid="user",
            target_uuid=f"topic-{i}",
            valid_at="2026-11-02",
        )
        for i in range(count)
    ]
    return episodes, facts


async def test_grouping_keeps_more_than_180_facts_even_when_model_omits_most_ids():
    episodes, facts = archive()
    llm = SimpleNamespace(
        generate_response=AsyncMock(
            return_value={
                "groups": [
                    {"title": "Detalles personales", "ids": ["fact-0000"]},
                    {"title": "No existe", "ids": ["invented"]},
                ]
            }
        )
    )
    compiler = HydrationCompiler(llm)
    candidates = compiler.fact_candidates(episodes, facts)
    grouped = await compiler.group(candidates)
    result, retained = compiler.render(grouped, facts, GraphStats(241, 240))
    assert len(retained) == 240
    assert set(result.metadata.included_edge_ids) == {f["uuid"] for f in facts}
    assert all(f["fact"] in result.compilation_text for f in facts)
    assert "No existe" not in result.compilation_text
    assert llm.generate_response.await_count > 1
    # Batches split work, never reduce the candidate set.
    offered = [
        fact
        for call in llm.generate_response.await_args_list
        for fact in json.loads(call.args[0][1].content)["facts"]
    ]
    assert {f["id"] for f in offered} == {f["uuid"] for f in facts}


def test_candidates_keep_all_episode_details_without_a_quote_extraction_gate():
    episodes, facts = archive(45)
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(episodes, facts)
    assert len(candidates) == 45
    assert {c["quote"] for c in candidates} == {f["fact"] for f in facts}


def test_only_exact_duplicates_with_same_subject_and_temporal_fields_are_merged():
    episodes, facts = archive(1)
    first = facts[0]
    duplicate = {**first, "uuid": "duplicate"}
    changed_date = {**first, "uuid": "different-date", "valid_at": "2025-01-01"}
    changed_person = {**first, "uuid": "different-person", "source_uuid": "cousin"}
    changed_detail = {
        **first,
        "uuid": "different-detail",
        "fact": first["fact"].replace("1000", "1001"),
    }
    facts = [first, duplicate, changed_date, changed_person, changed_detail]
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(episodes, facts)
    result, retained = compiler.render(candidates, facts, GraphStats(2, 5))
    assert len(retained) == 4
    assert set(result.metadata.included_edge_ids) == {f["uuid"] for f in facts}
    assert "2025-01-01" in result.compilation_text
    assert changed_detail["fact"] in result.compilation_text


def test_active_facts_are_kept_regardless_of_episode_words_and_retired_facts_stay_out():
    episodes = [
        dict(
            uuid="test",
            source="message",
            content="User: Prueba técnica de integración: no guardes esto.",
            valid_at="2026-11-02",
        ),
        dict(
            uuid="real",
            source="message",
            content="User: Mi prima Julia estudia ingeniería.",
            valid_at="2026-11-02",
        ),
    ]
    facts = [
        dict(uuid="test-fact", fact="Qwen está desplegado", episodes=["test"]),
        dict(uuid="real-fact", fact="Mi prima Julia estudia ingeniería.", episodes=["real"]),
        dict(uuid="old-fact", fact="Julia es mi hija", episodes=["real"], active=False),
        dict(uuid="orphan", fact="Julia vive en Bogotá.", episodes=[]),
    ]
    candidates = HydrationCompiler(None).fact_candidates(episodes, facts)
    assert {c["id"] for c in candidates} == {"test-fact", "real-fact", "orphan"}


async def test_incremental_grouping_preserves_all_unchanged_details_and_only_classifies_delta():
    episodes, facts = archive(220)
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(episodes, facts)
    hashes = {c["id"]: digest(c) for c in candidates}
    previous = dict(
        compiler=COMPILER_VERSION,
        selected=[{**c, "fact_section": "Historia detallada"} for c in candidates],
        seen_hashes=hashes,
        fact_sections={c["id"]: "Historia detallada" for c in candidates},
    )
    store = SimpleNamespace(current=AsyncMock(return_value=previous), group_cache=lambda job: None)
    service = PersistedHydration(store, compiler, None)
    compiler.group = AsyncMock(
        side_effect=lambda items, **kwargs: [{**c, "fact_section": "Nuevo proyecto"} for c in items]
    )
    new = {
        **candidates[0],
        "id": "new",
        "fact_id": "new",
        "quote": "Mi proyecto empieza el 15 de diciembre.",
    }
    grouped, _ = await service._group_incremental(
        {"user": "owner", "generation": 1}, candidates + [new]
    )
    assert len(grouped) == 221
    assert [c["id"] for c in compiler.group.await_args.args[0]] == ["new"]
    assert all(c["fact_section"] == "Historia detallada" for c in grouped if c["id"] != "new")


def test_budget_is_the_only_size_cutoff_and_never_truncates_a_retained_fact():
    episodes, facts = archive(20)
    compiler = HydrationCompiler(None, char_budget=1200)
    candidates = compiler.fact_candidates(episodes, facts)
    result, retained = compiler.render(candidates, facts, GraphStats(21, 20))
    assert 0 < len(retained) < 20
    assert len(result.compilation_text) <= 1200
    assert all(c["quote"] in result.compilation_text for c in retained)
    assert set(result.metadata.included_edge_ids) == {c["fact_id"] for c in retained}


async def test_large_archive_resumes_cached_grouping_without_reducing_or_repaying_batches():
    episodes, facts = archive(350)
    saved = {}

    async def rows(query, **params):
        if "RETURN n.sections" in query:
            return [{"sections": saved[params["key"]]}] if params["key"] in saved else []
        saved[params["key"]] = params["sections"]
        return [{"key": params["key"]}]

    async def classify(messages, **kwargs):
        batch = json.loads(messages[1].content)["facts"]
        return {"groups": [{"title": "Historia detallada", "ids": [c["id"] for c in batch]}]}

    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=classify))
    compiler = HydrationCompiler(llm)
    candidates = compiler.fact_candidates(episodes, facts)
    store = SimpleNamespace(rows=rows)
    for _ in range(40):
        try:
            grouped = await compiler.group(candidates, cache=GroupCache(store, {"user": "owner"}))
            break
        except WorkPending:
            pass
    else:
        raise AssertionError("Bounded grouping never finished")
    assert len(grouped) == 350
    assert llm.generate_response.await_count == len(saved)
    assert len(saved) >= 4


@pytest.mark.parametrize("mutation", ["deleted_alias", "changed_subject", "changed_detail"])
async def test_duplicate_coverage_revokes_snapshot_if_any_covered_record_changes(mutation):
    rows = [
        dict(
            uuid=fid,
            fact="Julia es mi prima.",
            source_uuid="user",
            target_uuid="julia",
            valid_at="2026-01-01",
            invalid_at=None,
        )
        for fid in ["primary", "duplicate"]
    ]
    result = HydrationResult(
        "Julia es mi prima.",
        CompilationMetadata(True, 5, compilation_revision="rev"),
        GraphStats(2, 2),
    )
    store = HydrationStore(None)
    store.current = AsyncMock(
        return_value=dict(
            compiler=COMPILER_VERSION,
            stamp="stamp",
            result=asdict(result),
            source_hashes={},
            edge_hashes={r["uuid"]: fact_hash(r) for r in rows},
        )
    )
    store.stamp = AsyncMock(return_value="stamp")
    store.rows = AsyncMock(side_effect=[[], rows])
    store.discard = AsyncMock()
    assert await store.read("owner")
    changed = [dict(r) for r in rows]
    if mutation == "deleted_alias":
        changed.pop()
    elif mutation == "changed_subject":
        changed[1]["source_uuid"] = "someone-else"
    else:
        changed[1]["fact"] = "Julia es mi hermana."
    store.rows.side_effect = [[], changed]
    assert await store.read("owner") is None
    store.discard.assert_awaited_once_with("owner", "rev")


def test_budget_prioritizes_recently_shared_biography_over_event_dates_and_unknown_dates():
    facts = [
        dict(
            uuid="biography",
            fact="En 2019 estudié química en el CES.",
            episodes=["recent"],
            valid_at="2019-01-01",
        ),
        dict(
            uuid="event",
            fact="Ayer compré una libreta azul.",
            episodes=["older"],
            valid_at="2026-11-02",
        ),
        dict(uuid="unknown", fact="Un recuerdo de procedencia desconocida.", episodes=[]),
    ]
    episodes = [
        dict(uuid="recent", valid_at="2026-11-02"),
        dict(uuid="older", valid_at="2026-10-01"),
    ]
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(episodes, facts)
    full, _ = compiler.render([candidates[0]], facts, GraphStats(1, 3))
    result, retained = HydrationCompiler(None, char_budget=len(full.compilation_text)).render(
        candidates, facts, GraphStats(1, 3)
    )
    assert [c["id"] for c in retained] == ["biography"]
    assert "2019" in result.compilation_text


def test_duplicate_primary_id_does_not_depend_on_database_row_order():
    episodes, facts = archive(1)
    facts.append({**facts[0], "uuid": "duplicate"})
    compiler = HydrationCompiler(None)
    assert compiler.fact_candidates(episodes, facts) == compiler.fact_candidates(
        episodes, list(reversed(facts))
    )


async def test_new_conversation_appends_grouping_work_without_reclassifying_cached_archive():
    episodes, facts = archive(1000)
    saved = {}
    cache = SimpleNamespace(load=AsyncMock(side_effect=lambda key: saved.get(key)))

    async def save(key, sections):
        saved[key] = sections

    async def classify(messages, **kwargs):
        batch = json.loads(messages[1].content)["facts"]
        return {"groups": [{"title": "Historia personal", "ids": [c["id"] for c in batch]}]}

    cache.save = save
    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=classify))
    compiler = HydrationCompiler(llm)
    candidates = compiler.fact_candidates(episodes, facts)
    await compiler.group(candidates, cache=cache)
    first_calls = llm.generate_response.await_count
    new = {
        **candidates[0],
        "id": "000-new",
        "fact_id": "000-new",
        "date": "2026-11-03",
        "quote": "Mi nuevo proyecto empieza mañana.",
    }
    grouped = await compiler.group([new] + candidates, cache=cache)
    assert len(grouped) == 1001
    assert llm.generate_response.await_count == first_calls + 1


async def test_missing_titles_can_be_retried_on_a_later_generation_without_losing_facts():
    episodes, facts = archive(1)
    saved = {}
    cache = SimpleNamespace(load=AsyncMock(side_effect=lambda key: saved.get(key)))

    async def save(key, sections):
        saved[key] = sections

    cache.save = save
    llm = SimpleNamespace(
        generate_response=AsyncMock(
            side_effect=[
                {"groups": []},
                {"groups": [{"title": "Detalles personales", "ids": [facts[0]["uuid"]]}]},
            ]
        )
    )
    compiler = HydrationCompiler(llm)
    candidates = [
        {**c, "classification_retry": True} for c in compiler.fact_candidates(episodes, facts)
    ]
    first = await compiler.group(candidates, cache=cache, retry_epoch=1)
    assert first[0]["classification_fallback"]
    again = await compiler.group(candidates, cache=cache, retry_epoch=1)
    assert again[0]["quote"] == first[0]["quote"]
    assert llm.generate_response.await_count == 1
    later = await compiler.group(candidates, cache=cache, retry_epoch=2)
    assert later[0]["fact_section"] == "Detalles personales"
    assert not later[0]["classification_fallback"]
    assert llm.generate_response.await_count == 2


def test_global_budget_keeps_corrections_ahead_of_many_old_topics():
    episodes, facts = archive(120)
    compiler = HydrationCompiler(None)
    grouped = compiler.fact_candidates(episodes, facts)
    for i, c in enumerate(grouped):
        c.update(
            correction=i < 60,
            fact_section="Familia" if i < 60 else f"Tema {i % 30}",
            date="2026-11-02" if i < 60 else "2015-01-01",
        )
    only_corrections, _ = compiler.render(grouped[:60], facts, GraphStats(121, 120))
    result, retained = HydrationCompiler(
        None, char_budget=len(only_corrections.compilation_text)
    ).render(grouped, facts, GraphStats(121, 120))
    assert {c["id"] for c in retained} == {c["id"] for c in grouped[:60]}
    assert len(result.compilation_text) <= len(only_corrections.compilation_text)


@pytest.mark.parametrize("mutation", ["delete", "move"])
async def test_mid_archive_changes_do_not_invalidate_all_grouping_batches(mutation):
    episodes, facts = archive(1000)
    saved = {}
    cache = SimpleNamespace(load=AsyncMock(side_effect=lambda key: saved.get(key)))

    async def save(key, sections):
        saved[key] = sections

    async def classify(messages, **kwargs):
        return {
            "groups": [
                {
                    "title": "Historia personal",
                    "ids": [c["id"] for c in json.loads(messages[1].content)["facts"]],
                }
            ]
        }

    cache.save = save
    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=classify))
    compiler = HydrationCompiler(llm)
    candidates = compiler.fact_candidates(episodes, facts)
    await compiler.group(candidates, cache=cache, retry_epoch=1)
    first_calls = llm.generate_response.await_count
    changed = [dict(c) for c in candidates]
    if mutation == "delete":
        changed.pop(5)
    else:
        changed[5]["date"] = "2026-11-04"
    grouped = await compiler.group(changed, cache=cache, retry_epoch=2)
    assert len(grouped) == len(changed)
    assert llm.generate_response.await_count <= first_calls + 2


async def test_published_payload_keeps_only_hashes_not_source_text():
    result = HydrationResult(
        "Julia es mi prima.",
        CompilationMetadata(True, 5, included_edge_ids=["fact"], included_episode_ids=["ep"]),
        GraphStats(2, 1),
    )
    store = HydrationStore(None)
    store.rows = AsyncMock(side_effect=[[{"revision": "ok"}], []])
    assert await store.publish(
        {"user": "owner", "token": "lease", "generation": 1},
        "stamp",
        result,
        [dict(uuid="ep", content="No guardes este mensaje: private test contents")],
        [dict(uuid="fact", fact="Julia es mi prima.")],
        seen_hashes={},
        fact_sections={},
    )
    payload = store.rows.await_args_list[0].kwargs["payload"]
    assert "private test contents" not in payload
    assert '"selected"' not in payload
