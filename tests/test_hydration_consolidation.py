import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import HydrationCompiler, WorkPending
from app.services.hydration_consolidation import consolidate
from app.services.hydration_result import GraphStats, source_facts
from app.services.hydration_store import GroupCache


def facts():
    return [
        dict(
            id=f"f{i}",
            fact_id=f"f{i}",
            duplicate_ids=[],
            quote=f"Leidy siente que aprender {field} solo tiene sentido si realiza un pregrado formal.",
            source_uuid="leidy",
            target_uuid=field,
            date="2026-11-02T10:00:00Z",
            valid_at=None,
            invalid_at=None,
            correction=False,
            episode_ids=["ep"],
            fact_section="Estudios",
        )
        for i, field in enumerate(["biología", "aeronáutica", "física"], 1)
    ]


def client(approved=True, text=None):
    text = (
        text
        or "Leidy siente que aprender biología, aeronáutica y física solo tiene sentido si realiza un pregrado formal."
    )
    return SimpleNamespace(
        generate_response=AsyncMock(
            side_effect=[
                {"records": [{"text": text, "ids": ["F1", "F2", "F3"]}]},
                {
                    "reviews": [
                        {
                            "record_index": 0,
                            "complete": approved,
                            "supported": approved,
                            "reason": "",
                        }
                    ]
                },
            ]
        )
    )


async def test_verified_consolidation_covers_all_sources_and_preserves_programmatic_dates():
    original = facts()
    llm = client()
    result = await consolidate(original, llm)
    assert len(result) == 1 and len(source_facts(result[0])) == 3
    edges = [
        {"uuid": f["id"], "fact": f["quote"], "valid_at": None, "invalid_at": None}
        for f in original
    ]
    rendered, retained = HydrationCompiler(None).render(result, edges, GraphStats(4, 3))
    assert set(rendered.metadata.included_edge_ids) == {"f1", "f2", "f3"}
    assert "2026-11-02T10:00:00Z" in rendered.compilation_text
    assert "biología, aeronáutica y física" in rendered.compilation_text
    assert llm.generate_response.await_count == 2


@pytest.mark.parametrize("issue", ["unsupported", "unknown_id", "duplicate_ids"])
async def test_suspicious_proposals_keep_original_details_and_coverage(issue):
    original = facts()
    llm = client(approved=False)
    proposal = {
        "records": [
            {
                "text": "Leidy siente que aprender biología, aeronáutica y física solo tiene sentido si realiza un pregrado formal.",
                "ids": ["F1", "F2", "F3"],
            }
        ]
    }
    if issue == "unknown_id":
        proposal["records"][0]["ids"].append("foreign")
    if issue == "duplicate_ids":
        proposal["records"].append(proposal["records"][0])
    llm.generate_response.side_effect = [
        proposal,
        {
            "reviews": [
                {
                    "record_index": 0,
                    "complete": issue != "unsupported",
                    "supported": issue != "unsupported",
                    "reason": "missing or unsupported content",
                }
            ]
        },
    ]
    result = await consolidate(original, llm)
    assert [f["quote"] for f in result] == [f["quote"] for f in original]


@pytest.mark.parametrize(
    "difference", ["source_uuid", "date", "valid_at", "invalid_at", "correction"]
)
async def test_different_subjects_or_temporal_states_are_not_fused(difference):
    original = facts()[:2]
    original[1][difference] = True if difference == "correction" else "different"
    llm = client()
    assert await consolidate(original, llm) == original
    llm.generate_response.assert_not_awaited()


async def test_failed_verification_keeps_literals_and_cache_avoids_repeated_cost():
    saved = {}
    cache = SimpleNamespace(load=AsyncMock(side_effect=lambda key, **kwargs: saved.get(key)))

    async def save(key, value, **kwargs):
        saved[key] = value

    cache.save = save
    llm = client(False)
    original = facts()
    assert await consolidate(original, llm, cache) == original
    assert await consolidate(original, llm, cache) == original
    assert llm.generate_response.await_count == 2


async def test_provider_failure_does_not_replace_memory_with_empty_or_approximate_text():
    original = facts()
    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=TimeoutError()))
    assert await consolidate(original, llm) == original


async def test_editing_any_source_changes_cache_key_and_rechecks_the_rewrite():
    saved = {}
    cache = SimpleNamespace(load=AsyncMock(side_effect=lambda key, **kwargs: saved.get(key)))

    async def save(key, value, **kwargs):
        saved[key] = value

    cache.save = save
    original = facts()
    llm = client()
    assert len(await consolidate(original, llm, cache)) == 1
    changed = [dict(f) for f in original]
    changed[0]["quote"] = changed[0]["quote"].replace("biología", "biología molecular")
    llm.generate_response.side_effect = [{"records": []}]
    assert await consolidate(changed, llm, cache) == changed
    assert llm.generate_response.await_count == 3


async def test_budget_falls_back_to_whole_original_facts_if_merged_record_does_not_fit():
    original = facts()
    merged = await consolidate(original, client())
    assert len(merged) == 1
    edges = [dict(uuid=f["id"], fact=f["quote"], valid_at=None, invalid_at=None) for f in original]
    full, _ = HydrationCompiler(None).render(merged, edges, GraphStats(4, 3))
    compiler = HydrationCompiler(None, char_budget=len(full.compilation_text) - 5)
    result, retained = compiler.render(merged, edges, GraphStats(4, 3))
    assert retained and len(result.compilation_text) <= compiler.char_budget
    assert all(f["quote"] in result.compilation_text for f in retained)
    assert set(result.metadata.included_edge_ids) == {f["fact_id"] for f in retained}


async def test_changed_member_or_alias_cannot_remain_hidden_inside_verified_text():
    original = facts()
    original[0]["duplicate_ids"] = ["alias"]
    merged = await consolidate(original, client())
    edges = [dict(uuid=f["id"], fact=f["quote"], valid_at=None, invalid_at=None) for f in original]
    edges.append({**edges[0], "uuid": "alias"})
    compiler = HydrationCompiler(None)
    result, _ = compiler.render(merged, edges, GraphStats(4, 4))
    assert set(result.metadata.included_edge_ids) == {"f1", "f2", "f3", "alias"}
    edges[-1]["fact"] = "Dato corregido"
    result, _ = compiler.render(merged, edges, GraphStats(4, 4))
    assert set(result.metadata.included_edge_ids) == {"f2", "f3"}
    assert "biología" not in result.compilation_text


@pytest.mark.parametrize(
    "reviews", [[], [dict(record_index=0, complete=True, supported=True, reason="")] * 2]
)
async def test_missing_or_duplicate_verdict_is_not_approval(reviews):
    llm = client()
    proposal = llm.generate_response.side_effect
    first = next(proposal)
    llm.generate_response.side_effect = [first, {"reviews": reviews}]
    assert await consolidate(facts(), llm) == facts()


async def test_grouping_and_consolidation_share_two_call_budget():
    store = SimpleNamespace(rows=AsyncMock(return_value=[]))
    cache = GroupCache(store, {"user": "owner"})
    assert await cache.load("group") is None
    with pytest.raises(WorkPending):
        await consolidate(facts(), client(), cache)
    assert cache.calls == 1
    fresh = GroupCache(store, {"user": "owner"})
    assert await fresh.load("consolidation", calls=2) is None
    with pytest.raises(WorkPending):
        await fresh.load("another")


async def test_provider_failure_is_cached_for_job_but_retried_after_new_ingestion():
    saved = {}

    async def load(key, *, retry_epoch, **kwargs):
        record = saved.get(key)
        return record[0] if record and record[1] in (None, retry_epoch) else None

    async def save(key, value, *, retry_epoch=None):
        saved[key] = (value, retry_epoch)

    cache = SimpleNamespace(load=load, save=save)
    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=TimeoutError()))
    assert await consolidate(facts(), llm, cache, retry_epoch=1) == facts()
    assert await consolidate(facts(), llm, cache, retry_epoch=1) == facts()
    assert llm.generate_response.await_count == 1
    llm = client()
    assert len(await consolidate(facts(), llm, cache, retry_epoch=2)) == 1
    assert llm.generate_response.await_count == 2
    assert len(await consolidate(facts(), llm, cache, retry_epoch=3)) == 1
    assert llm.generate_response.await_count == 2


async def test_consolidation_timeout_keeps_literal_memory(monkeypatch):
    monkeypatch.setattr("app.services.hydration_result.LLM_TIMEOUT_SECONDS", 0.001)

    async def delayed(*args, **kwargs):
        await asyncio.sleep(10)

    llm = SimpleNamespace(generate_response=AsyncMock(side_effect=delayed))
    assert await consolidate(facts(), llm) == facts()
    assert llm.generate_response.await_count == 1


async def test_parallel_english_enumeration_keeps_language_and_all_items():
    original = [
        {**f, "quote": f"Leidy experiences a physical reaction when seeing {insect}."}
        for f, insect in zip(facts(), ["bees", "ants"], strict=False)
    ]
    llm = client()
    llm.generate_response.side_effect = [
        {
            "records": [
                {
                    "text": "Leidy experiences a physical reaction when seeing bees and ants.",
                    "ids": ["F1", "F2"],
                }
            ]
        },
        {"reviews": [{"record_index": 0, "complete": True, "supported": True, "reason": ""}]},
    ]
    result = await consolidate(original, llm)
    assert len(result) == 1 and "bees and ants" in result[0]["quote"]
    assert {f["id"] for f in source_facts(result[0])} == {"f1", "f2"}


async def test_model_can_paraphrase_when_review_accepts_without_word_rules():
    llm = client(
        text="Para Leidy, un pregrado formal da sentido al aprendizaje de física, biología y aeronáutica."
    )
    result = await consolidate(facts(), llm)
    assert len(result) == 1
    assert result[0]["quote"].startswith("Para Leidy")
    assert {f["id"] for f in source_facts(result[0])} == {"f1", "f2", "f3"}


async def test_empty_proposal_keeps_every_original_without_forcing_consolidation():
    llm = SimpleNamespace(generate_response=AsyncMock(return_value={"records": []}))
    assert await consolidate(facts(), llm) == facts()
    assert llm.generate_response.await_count == 1


async def test_partial_proposal_keeps_every_unmerged_original():
    llm = client()
    llm.generate_response.side_effect = [
        {
            "records": [
                {
                    "text": "Leidy valora estudiar biología y aeronáutica mediante un pregrado formal.",
                    "ids": ["F1", "F2"],
                }
            ]
        },
        {"reviews": [{"record_index": 0, "complete": True, "supported": True, "reason": ""}]},
    ]
    result = await consolidate(facts(), llm)
    assert len(result) == 2
    assert result[-1] == facts()[-1]
    assert {f["id"] for record in result for f in source_facts(record)} == {"f1", "f2", "f3"}
