from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import HydrationCompiler
from app.services.hydration_consolidation import apply_plan, prioritize
from app.services.hydration_consolidation import organize as group_facts
from app.services.hydration_result import GraphStats


async def organize(facts, llm):
    priorities, proposals = await prioritize(facts, llm)
    plan = await group_facts(facts, llm, proposals)
    return {**plan, "priority_ids": priorities}


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
                    "records": records
                    if records is not None
                    else [
                        {"text": "Le gustan el café y el té por la mañana.", "ids": ["F1", "F2"]}
                    ],
                },
                {
                    "groups": [{"title": "Preferencias", "ids": ["F1", "F2"]}],
                    "reviews": reviews
                    if reviews is not None
                    else [
                        {
                            "record_index": 0,
                            "complete": True,
                            "supported": True,
                            "reason": "Conserva ambos.",
                        }
                    ],
                },
            ]
        )
    )


async def test_one_organization_and_one_review_cover_all_originals_with_dates():
    llm = provider()
    source = facts()
    plan = await organize(source, llm)
    assert llm.generate_response.await_count == 2
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
        "hydration.prioritize",
        "hydration.organize",
    ]


@pytest.mark.parametrize("ids", [["F1", "fake"], ["F1", "F1"], ["F1"], []])
async def test_invalid_proposal_ids_do_not_suppress_any_literal_or_trigger_review(ids):
    llm = provider(records=[{"text": "rewrite", "ids": ids}])
    source = facts()
    plan = await organize(source, llm)
    assert llm.generate_response.await_count == 2
    assert [f["quote"] for f in apply_plan(source, plan)] == [f["quote"] for f in source]
    import json

    assert "records" not in json.loads(llm.generate_response.await_args.args[0][-1].content)


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
    plan = await organize(source, llm)
    assert (not plan["records"]) and (llm.generate_response.await_count == 2)
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
    plan = await organize(source, provider(reviews=reviews))
    assert len(apply_plan(source, plan)) == 2
    assert not plan["records"]


@pytest.mark.parametrize("failed_call", [1, 2])
async def test_provider_failure_publishes_literals_with_literal_facts(failed_call):
    llm = provider()
    responses = list(llm.generate_response.side_effect)
    responses[failed_call - 1] = RuntimeError("provider unavailable")
    llm.generate_response.side_effect = responses
    source = facts()
    plan = await organize(source, llm)
    assert llm.generate_response.await_count == 2
    assert [f["quote"] for f in apply_plan(source, plan)] == [f["quote"] for f in source]
    assert bool(plan["sections"]) == (failed_call == 1)


async def test_grouping_failure_preserves_the_first_calls_priorities():
    llm = provider(records=[])
    llm.generate_response.side_effect = [
        {"priority_ids": ["F2", "F1"], "records": []},
        RuntimeError("grouping unavailable"),
    ]
    plan = await organize(facts(), llm)
    assert plan == {"priority_ids": ["1", "0"], "sections": {}, "records": []}
    assert llm.generate_response.await_count == 2


async def test_grouping_cannot_approve_a_merge_across_different_themes():
    llm = provider()
    responses = list(llm.generate_response.side_effect)
    responses[1]["groups"] = [
        {"title": "Primera preferencia", "ids": ["F1"]},
        {"title": "Segunda preferencia", "ids": ["F2"]},
    ]
    llm.generate_response.side_effect = responses
    plan = await organize(facts(), llm)
    assert not plan["records"]
    assert len(apply_plan(facts(), plan)) == 2
    assert llm.generate_response.await_count == 2


async def test_proposal_with_a_source_outside_the_core_is_not_reviewed_or_covered():
    import json

    source = facts()
    llm = provider()
    _, proposals = await prioritize(source, llm)
    plan = await group_facts(source[:1], llm, proposals)
    assert not plan["records"]
    assert set(plan["sections"]) == {"0"}
    sent = json.loads(llm.generate_response.await_args.args[0][-1].content)
    assert "records" not in sent
    assert len(sent["facts"]) == 1


async def test_no_proposals_or_missing_topics_keeps_every_fact_within_two_calls():
    source = facts()
    llm = provider(records=[])
    llm.generate_response.side_effect = [{"records": []}, {"groups": []}]
    plan = await organize(source, llm)
    assert len(apply_plan(source, plan)) == 2
    assert llm.generate_response.await_count == 2
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
    plan = await organize(facts(), llm)
    assert apply_plan(facts(), plan)[0]["quote"] == "Su rutina matutina incluye tomar café o té."


async def test_partial_proposal_preserves_every_unmerged_fact():
    source = facts() + [{**facts()[0], "id": "2", "fact_id": "2", "quote": "Tiene un gato."}]
    plan = await organize(source, provider())
    records = apply_plan(source, plan)
    assert len(records) == 2 and records[-1]["quote"] == "Tiene un gato."


async def test_changed_source_is_not_hidden_inside_verified_paraphrase():
    source = facts()
    plan = await organize(source, provider())
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
    plan = await organize(source, llm)
    rows = [
        dict(uuid=f["id"], fact=f["quote"], valid_at=f["valid_at"], invalid_at=None) for f in source
    ]
    result, _ = HydrationCompiler(None, char_budget=1300).render(
        apply_plan(source, plan), rows, GraphStats(3, 2)
    )
    assert set(result.metadata.included_edge_ids) == {"0", "1"}
    assert "café por la mañana." in result.compilation_text
    assert len(result.compilation_text) <= 1300


async def test_semantic_priority_places_older_personal_detail_before_connected_recent_noise():
    compiler = HydrationCompiler(None, char_budget=1100)
    rows = [
        dict(
            uuid="personal",
            fact="Necesita apoyo para organizar las comidas.",
            source_uuid="user",
            target_uuid="routine",
            priority=3,
            date="2020-01-01",
        ),
        *[
            dict(
                uuid=f"noise{i}",
                fact="Dato enciclopédico " + str(i) + " " + ("x" * 100),
                source_uuid="user",
                target_uuid=str(i),
                priority=1,
                date="2026-11-01",
            )
            for i in range(10)
        ],
    ]
    source = compiler.fact_candidates(rows)
    # Aliases follow canonical UUID order, independently of graph ranking.
    aliases = {f["id"]: f"F{i}" for i, f in enumerate(sorted(source, key=lambda f: f["id"]), 1)}
    llm = SimpleNamespace(
        generate_response=AsyncMock(
            return_value={
                "groups": [],
                "records": [],
                "priority_ids": [aliases["personal"]],
            }
        )
    )
    plan = await organize(source, llm)
    result, _ = compiler.render(apply_plan(source, plan), rows, GraphStats(12, 11))
    assert result.metadata.included_edge_ids[0] == "personal"
    assert "Necesita apoyo para organizar las comidas." in result.compilation_text
    assert len(result.compilation_text) <= 1100
    assert len(result.metadata.included_edge_ids) < len(rows)


async def test_unknown_duplicate_and_missing_priority_ids_do_not_remove_literal_facts():
    source = facts()
    llm = SimpleNamespace(
        generate_response=AsyncMock(
            return_value={
                "groups": [],
                "records": [],
                "priority_ids": ["fake", "F2", "F2"],
            }
        )
    )
    plan = await organize(source, llm)
    assert plan["priority_ids"] == ["1"]
    records = apply_plan(source, plan)
    assert {r["id"] for r in records} == {"0", "1"}
    assert next(r for r in records if r["id"] == "1")["semantic_rank"] == 0
    assert next(r for r in records if r["id"] == "0")["semantic_rank"] == 0.25


def test_merged_record_inherits_its_highest_priority_member():
    source = facts()
    records = apply_plan(
        source,
        {
            "priority_ids": ["1", "0"],
            "records": [{"ids": ["0", "1"], "text": "Le gustan el café y el té por la mañana."}],
        },
    )
    assert len(records) == 1 and records[0]["semantic_rank"] == 0
    assert {f["id"] for f in records[0]["source_facts"]} == {"0", "1"}


def test_unranked_early_graph_fact_can_reach_core_when_budget_is_tight():
    compiler = HydrationCompiler(None, char_budget=1100)
    rows = [
        dict(
            uuid="early",
            fact="Necesita ayuda práctica para preparar comidas.",
            source_uuid="user",
            target_uuid="routine",
            priority=1,
            date="2026-11-02",
        )
    ]
    rows += [
        dict(
            uuid=f"ranked{i:02}",
            fact="Detalle personal " + str(i) + " " + ("x" * 110),
            source_uuid="user",
            target_uuid=str(i),
            priority=1,
            date="2026-11-01",
        )
        for i in range(20)
    ]
    source = compiler.fact_candidates(rows)
    plan = {"priority_ids": [f"ranked{i:02}" for i in range(20)]}
    result, _ = compiler.render(apply_plan(source, plan), rows, GraphStats(22, 21))
    assert len(result.compilation_text) <= 1100
    assert "early" in result.metadata.included_edge_ids
    assert len(result.metadata.included_edge_ids) < len(rows)
    assert set(result.metadata.included_edge_ids) <= {f["id"] for f in source}
