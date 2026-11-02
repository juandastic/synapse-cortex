from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import HydrationCompiler
from app.services.hydration_consolidation import apply_plan, organize
from app.services.hydration_result import GraphStats


def facts():
    return HydrationCompiler(None).fact_candidates(
        [
            dict(
                uuid=str(i),
                fact=text,
                source_uuid="user",
                target_uuid=str(i),
                valid_at="2026-01-01",
                invalid_at=None,
                date="2026-11-02",
                episode_ids=["ep"],
            )
            for i, text in enumerate(
                ["Le gusta el café por la mañana.", "Le gusta el té por la mañana."]
            )
        ]
    )


def provider(records=None, reviews=None):
    return SimpleNamespace(
        generate_response=AsyncMock(
            side_effect=[
                {
                    "groups": [{"title": "Preferencias", "ids": ["F1", "F2"]}],
                    "records": records
                    if records is not None
                    else [
                        {"text": "Le gustan el café y el té por la mañana.", "ids": ["F1", "F2"]}
                    ],
                },
                {
                    "reviews": reviews
                    if reviews is not None
                    else [
                        {
                            "record_index": 0,
                            "complete": True,
                            "supported": True,
                            "reason": "Conserva ambos.",
                        }
                    ]
                },
            ]
        )
    )


async def test_one_organization_and_one_review_cover_all_originals_with_dates():
    llm = provider()
    source = facts()
    plan, reusable = await organize(source, llm)
    assert reusable and llm.generate_response.await_count == 2
    assert plan["records"][0]["ids"] == ["0", "1"]
    records = apply_plan(source, plan)
    edges = [
        dict(
            uuid=f["id"],
            fact=f["quote"],
            source_uuid=f["source_uuid"],
            target_uuid=f["target_uuid"],
            valid_at=f["valid_at"],
            invalid_at=f["invalid_at"],
        )
        for f in source
    ]
    result, _ = HydrationCompiler(None).render(records, edges, GraphStats(3, 2))
    assert "café y el té" in result.compilation_text
    assert set(result.metadata.included_edge_ids) == {"0", "1"}
    assert result.metadata.included_episode_ids == ["ep"]
    assert "2026-01-01" in result.compilation_text
    assert [c.kwargs["prompt_name"] for c in llm.generate_response.await_args_list] == [
        "hydration.organize",
        "hydration.verify",
    ]


@pytest.mark.parametrize("ids", [["F1", "fake"], ["F1", "F1"], ["F1"], []])
async def test_invalid_proposal_ids_do_not_suppress_any_literal_or_trigger_review(ids):
    llm = provider(records=[{"text": "rewrite", "ids": ids}])
    source = facts()
    plan, ok = await organize(source, llm)
    assert ok and llm.generate_response.await_count == 1
    assert [f["quote"] for f in apply_plan(source, plan)] == [f["quote"] for f in source]


@pytest.mark.parametrize(
    "difference",
    [
        {"source_uuid": "cousin"},
        {"valid_at": "2025-01-01"},
        {"invalid_at": "2027-01-01"},
        {"date": "2026-11-03"},
        {"correction": True},
    ],
)
async def test_different_subjects_and_temporal_scopes_stay_separate(difference):
    source = facts()
    source[1].update(difference)
    llm = provider()
    plan, ok = await organize(source, llm)
    assert ok and (not plan["records"]) and (llm.generate_response.await_count == 1)
    assert len(apply_plan(source, plan)) == 2


@pytest.mark.parametrize(
    "reviews",
    [
        [],
        [{"record_index": 0, "complete": False, "supported": True, "reason": "Missing detail"}],
        [{"record_index": 0, "complete": True, "supported": False, "reason": "Inference"}],
        [{"record_index": 0, "complete": True, "supported": True, "reason": ""}] * 2,
    ],
)
async def test_absent_rejected_or_duplicate_verdict_keeps_originals(reviews):
    source = facts()
    plan, ok = await organize(source, provider(reviews=reviews))
    assert ok and len(apply_plan(source, plan)) == 2
    assert not plan["records"]


@pytest.mark.parametrize("failed_call", [1, 2])
async def test_provider_failure_publishes_literals_without_reusable_fingerprint(failed_call):
    llm = provider()
    responses = list(llm.generate_response.side_effect)
    responses[failed_call - 1] = RuntimeError("provider unavailable")
    llm.generate_response.side_effect = responses
    source = facts()
    plan, reusable = await organize(source, llm)
    assert not reusable and llm.generate_response.await_count == failed_call
    assert [f["quote"] for f in apply_plan(source, plan)] == [f["quote"] for f in source]


async def test_no_proposals_or_missing_topics_keeps_every_fact_without_extra_call():
    source = facts()
    llm = provider(records=[])
    llm.generate_response.side_effect = [{"groups": [], "records": []}]
    plan, ok = await organize(source, llm)
    assert ok and len(apply_plan(source, plan)) == 2
    assert llm.generate_response.await_count == 1
    assert all(
        (
            f["quote"] == original["quote"]
            for f, original in zip(apply_plan(source, plan), source, strict=True)
        )
    )


async def test_model_can_paraphrase_without_linguistic_word_rules():
    llm = provider(
        records=[{"text": "Su rutina matutina incluye tomar café o té.", "ids": ["F1", "F2"]}]
    )
    plan, ok = await organize(facts(), llm)
    assert (
        ok
        and apply_plan(facts(), plan)[0]["quote"] == "Su rutina matutina incluye tomar café o té."
    )


async def test_partial_proposal_preserves_every_unmerged_fact():
    source = facts() + [{**facts()[0], "id": "2", "fact_id": "2", "quote": "Tiene un gato."}]
    plan, _ = await organize(source, provider())
    records = apply_plan(source, plan)
    assert len(records) == 2 and records[-1]["quote"] == "Tiene un gato."


async def test_changed_source_is_not_hidden_inside_verified_paraphrase():
    source = facts()
    plan, _ = await organize(source, provider())
    records = apply_plan(source, plan)
    rows = [
        dict(uuid=f["id"], fact=f["quote"], valid_at=f["valid_at"], invalid_at=None) for f in source
    ]
    rows[1]["fact"] = "Other detail"
    result, _ = HydrationCompiler(None).render(records, rows, GraphStats(3, 2))
    assert result.metadata.included_edge_ids == ["0"]
    assert "café y el té" not in result.compilation_text


async def test_merged_record_over_budget_falls_back_to_whole_sources():
    source = facts()
    llm = provider(records=[{"text": "x" * 3000, "ids": ["F1", "F2"]}])
    plan, _ = await organize(source, llm)
    rows = [
        dict(uuid=f["id"], fact=f["quote"], valid_at=f["valid_at"], invalid_at=None) for f in source
    ]
    result, _ = HydrationCompiler(None, char_budget=1300).render(
        apply_plan(source, plan), rows, GraphStats(3, 2)
    )
    assert set(result.metadata.included_edge_ids) == {"0", "1"}
    assert "café por la mañana." in result.compilation_text
    assert len(result.compilation_text) <= 1300
