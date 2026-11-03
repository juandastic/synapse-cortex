from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.hydration_compiler import COMPILER_VERSION, HydrationCompiler
from app.services.hydration_result import GraphStats
from app.services.hydration_store import HydrationStore
from app.services.persisted_hydration import PersistedHydration, UnsupportedArchive


def setup(edges=None):
    edges = (
        edges
        if edges is not None
        else [
            dict(
                uuid="f",
                fact="Julia es mi prima y estudia ingeniería.",
                source_uuid="user",
                target_uuid="Julia",
                date="2026-11-02",
                episode_ids=["ep"],
            )
        ]
    )
    store = SimpleNamespace(
        select=AsyncMock(return_value=(edges, "ep", GraphStats(2, len(edges)))),
        current=AsyncMock(return_value=None),
        validate=AsyncMock(return_value=True),
        head=AsyncMock(return_value="ep"),
        publish=AsyncMock(return_value=True),
    )
    llm = SimpleNamespace(generate_response=AsyncMock(return_value={"groups": [], "records": []}))
    service = PersistedHydration(store, HydrationCompiler(llm))
    return (service, store, llm)


async def test_every_build_runs_model_without_reading_previous_core():
    service, store, llm = setup()
    store.current.side_effect = AssertionError("Build must not load the previous core")
    job = dict(user="owner", token="lease", generation=1)
    for _ in range(3):
        assert await service.build(job)
    assert llm.generate_response.await_count == 3
    store.current.assert_not_awaited()


async def test_rebuild_uses_latest_selected_graph_facts_and_provenance():
    service, store, llm = setup()
    job = dict(user="owner", token="lease", generation=1)
    assert await service.build(job)
    rows, head, stats = store.select.return_value
    store.select.return_value = (
        [
            {
                **rows[0],
                "fact": "Julia estudia medicina.",
                "date": "2026-11-03",
                "episode_ids": ["new"],
            }
        ],
        head,
        stats,
    )
    assert await service.build(job)
    assert llm.generate_response.await_count == 2
    result = store.publish.await_args.args[1]
    assert "Julia estudia medicina." in result.compilation_text
    assert "2026-11-03" in result.compilation_text
    assert result.metadata.included_episode_ids == ["new"]


@pytest.mark.parametrize("failure", ["deps", "head"])
async def test_change_during_model_or_cache_preparation_does_not_publish(failure):
    service, store, _ = setup()
    if failure == "deps":
        store.validate.return_value = False
    else:
        store.head.return_value = "new"
    assert not await service.build(dict(user="owner", token="lease", generation=1))
    store.publish.assert_not_awaited()


async def test_provider_failure_preserves_details_and_retry_occurs_only_on_next_build():
    service, store, llm = setup()
    llm.generate_response.side_effect = RuntimeError("offline")
    assert await service.build(dict(user="owner", token="lease", generation=1))
    assert "Julia es mi prima" in store.publish.await_args.args[1].compilation_text
    assert llm.generate_response.await_count == 1


@pytest.mark.parametrize("head,stats", [("ep", GraphStats(0, 0)), (None, GraphStats(2, 0))])
async def test_episodes_or_graph_without_active_facts_preserve_classic_memory(head, stats):
    service, store, llm = setup([])
    store.select.return_value = ([], head, stats)
    with pytest.raises(UnsupportedArchive):
        await service.build(dict(user="owner"))
    store.publish.assert_not_awaited()
    llm.generate_response.assert_not_awaited()


async def test_empty_user_can_publish_empty_profile_without_model_call():
    service, store, llm = setup([])
    store.select.return_value = ([], None, GraphStats(0, 0))
    store.head.return_value = None
    assert await service.build(dict(user="owner"))
    assert store.publish.await_args.args[1].compilation_text == ""
    llm.generate_response.assert_not_awaited()


async def test_cache_failure_does_not_block_publication_and_dependency_check_is_after_cache():
    service, store, _ = setup()
    order = []

    async def cache(*args):
        order.append("cache")
        raise RuntimeError("offline")

    async def validate(*args):
        order.append("validate")
        return True

    service.cache_manager = SimpleNamespace(create_compilation_cache=cache)
    store.validate.side_effect = validate
    assert await service.build(dict(user="owner"))
    assert order == ["cache", "validate"]


async def test_publish_contains_only_result_and_dependencies_and_is_fenced():
    service, stub, _ = setup()
    await service.build(dict(user="owner"))
    call = stub.publish.await_args
    store = HydrationStore(None)
    store.rows = AsyncMock(return_value=[])
    assert not await store.publish(
        dict(user="owner", token="lease", generation=2), call.args[1], **call.kwargs
    )
    query = store.rows.await_args.args[0]
    params = store.rows.await_args.kwargs
    assert (
        "n.hydration_generation=$generation" in query
        and "n.hydration_lease_until>timestamp()" in query
    )
    assert "hydration_owner:$token" in query
    import json

    payload = json.loads(params["payload"])
    assert set(payload) == {"compiler", "deps", "head_episode", "result"}
    assert params["user"] == "owner" and params["generation"] == 2
    assert payload["result"] == asdict(call.args[1])


async def test_request_migrates_old_terminal_compiler_without_read_generation_race():
    store = HydrationStore(None)
    store.rows = AsyncMock()
    await store.request("owner")
    query = store.rows.await_args.args[0]
    assert "hydration_requested_compiler" in query
    assert store.rows.await_args.kwargs["compiler"] == COMPILER_VERSION
    assert not store.rows.await_args.kwargs["source_changed"]


async def test_release_preserves_newer_generation_retry_and_success_resets_attempts():
    store = HydrationStore(None)
    store.rows = AsyncMock()
    await store.release(dict(user="owner", token="lease", generation=1, attempts=3), failed=True)
    query = store.rows.await_args.args[0]
    assert "ELSE n.hydration_retry_at" in query
    assert "WHEN n.hydration_dirty=false THEN 0" in query
    assert store.rows.await_args.kwargs["delay"] == 240000


async def test_large_graph_needs_at_most_two_model_calls_after_input_caps():
    import json

    edges = [
        dict(
            uuid=f"f{i:04}",
            fact=f"Recuerdo {i} " + "x" * 250,
            source_uuid="user",
            target_uuid=str(i),
            date="2026-11-02",
        )
        for i in range(5000)
    ]
    service, store, llm = setup(edges)
    llm.generate_response.side_effect = [
        {"groups": [], "records": [{"text": "Recuerdos relacionados", "ids": ["F1", "F2"]}]},
        {"reviews": [{"record_index": 0, "complete": True, "supported": True, "reason": ""}]},
    ]
    assert await service.build(dict(user="owner"))
    assert llm.generate_response.await_count == 2
    sent = json.loads(llm.generate_response.await_args_list[0].args[0][-1].content)["facts"]
    assert len(sent) <= 800 and sum((len(f["text"]) for f in sent)) <= 120000
    assert len(store.publish.await_args.args[1].metadata.included_edge_ids) <= 800


async def test_core_can_be_deleted_and_regenerated_without_carrying_previous_decisions():
    import json

    service, store, llm = setup()
    llm.generate_response.return_value = {"priority_ids": ["F1"], "groups": [], "records": []}
    job = dict(user="owner", token="lease", generation=1)
    assert await service.build(job)
    first = store.publish.await_args.args[1]
    store.current.return_value = None
    assert await service.build(job)
    regenerated = store.publish.await_args.args[1]
    assert llm.generate_response.await_count == 2
    assert first.metadata.included_edge_ids == regenerated.metadata.included_edge_ids
    inputs = [json.loads(c.args[0][-1].content) for c in llm.generate_response.await_args_list]
    assert inputs[0] == inputs[1] and set(inputs[0]) == {"facts"}
