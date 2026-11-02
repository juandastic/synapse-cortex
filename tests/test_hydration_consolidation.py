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


@pytest.mark.parametrize(
    "issue", ["unsupported", "unknown_id", "duplicate_ids", "missing_number", "missing_name"]
)
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
    if issue == "missing_number":
        original[0]["quote"] = original[0]["quote"].replace("biología", "biología en 2027")
    if issue == "missing_name":
        proposal["records"][0]["text"] = proposal["records"][0]["text"].replace(
            "Leidy", "La usuaria"
        )
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

    async def save(key, value):
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

    async def save(key, value):
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


async def test_excluded_context_never_enters_consolidation_or_its_cache():
    original = facts()
    for fact in original:
        fact["excluded_context"] = [{"text": "No guardes este mensaje: privado"}]
    llm = client()
    assert await consolidate(original, llm) == original
    llm.generate_response.assert_not_awaited()


@pytest.mark.parametrize(
    "texts",
    [
        [
            "Una mariposa estaba al lado del nochero de Leidy el 17 de noviembre de 2025.",
            "Una mariposa estaba posada sobre la almohada de Leidy el 17 de noviembre de 2025.",
            "Una mariposa estaba en el cuarto de Leidy el 17 de noviembre de 2025.",
        ],
        [
            "El Assistant describe que la fisioterapia se divide en fases de movilidad temprana y activación isométrica basándose en la cicatrización del tejido.",
            "El Assistant le explica al User que la fisioterapia se divide en dos fases, incluyendo una primera de movilidad temprana y activación isométrica, según la biología de cicatrización.",
        ],
    ],
)
async def test_real_ambiguous_locations_and_phase_descriptions_stay_literal_even_if_model_would_approve(
    texts,
):
    original = [{**f, "quote": text} for f, text in zip(facts(), texts, strict=False)]
    llm = client()
    assert await consolidate(original, llm) == original
    llm.generate_response.assert_not_awaited()


async def test_lowercase_enumeration_item_cannot_disappear_behind_covered_id():
    original = [
        {**f, "quote": f"Al usuario le gusta tomar {drink} por la mañana antes del trabajo."}
        for f, drink in zip(facts(), ["café", "té"], strict=False)
    ]
    llm = client(text=original[0]["quote"])
    llm.generate_response.side_effect = [
        {"records": [{"text": original[0]["quote"], "ids": ["F1", "F2"]}]},
        {"reviews": [{"record_index": 0, "complete": True, "supported": True, "reason": ""}]},
    ]
    records = await consolidate(original, llm)
    assert records == original
    edges = [dict(uuid=f["id"], fact=f["quote"]) for f in original]
    result, _ = HydrationCompiler(None).render(records, edges, GraphStats(3, 2))
    assert "café" in result.compilation_text and "té" in result.compilation_text
    assert set(result.metadata.included_edge_ids) == {"f1", "f2"}
    assert llm.generate_response.await_count == 1


@pytest.mark.parametrize(
    "texts",
    [
        [
            "Una mariposa estaba en el nochero de Leidy el 17 de noviembre de 2025.",
            "Una mariposa estaba en la almohada de Leidy el 17 de noviembre de 2025.",
        ],
        [
            "Leidy dice que vive en Bogotá con su pareja desde hace varios años.",
            "Leidy dice que vive en Medellín con su pareja desde hace varios años.",
        ],
        [
            "Leidy dice que toma 50 mg de sertralina por la mañana desde hace varios meses.",
            "Leidy dice que toma 100 mg de sertralina por la mañana desde hace varios meses.",
        ],
        [
            "Leidy dice que le gusta mucho el café por la mañana.",
            "Leidy dice que no le gusta el café por la mañana.",
        ],
        [
            "Leidy acompañó el proceso de identificación neurodivergente de Emmanuel.",
            "Leidy acompañó el proceso de identificación neurodivergente de Wil.",
        ],
    ],
)
async def test_incompatible_locations_doses_negation_and_proper_name_slots_stay_literal(texts):
    original = [{**f, "quote": text} for f, text in zip(facts(), texts, strict=False)]
    llm = client()
    assert await consolidate(original, llm) == original
    llm.generate_response.assert_not_awaited()


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


@pytest.mark.parametrize(
    "text",
    [
        "Leidy siente que aprender biología y aeronáutica solo tiene sentido si realiza un pregrado formal.",
        "Leidy siente que aprender biología, aeronáutica y física solo tiene sentido si realiza un pregrado formal. Actualmente está matriculada.",
        "Leidy realiza un pregrado formal y siente que aprender biología, aeronáutica y física solo tiene sentido si.",
    ],
)
async def test_model_approval_cannot_override_missing_items_invented_words_or_changed_predicate(
    text,
):
    llm = client(text=text)
    assert await consolidate(facts(), llm) == facts()
    assert llm.generate_response.await_count == 1
