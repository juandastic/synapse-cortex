from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services.hydration_v2 import (
    FETCH_EDGES_QUERY,
    FETCH_EPISODES_QUERY,
    FETCH_NODES_QUERY,
    HydrationV2Engine,
)


def graph_driver(nodes=None, edges=None, episodes=None):
    records = {
        FETCH_NODES_QUERY: nodes or [],
        FETCH_EDGES_QUERY: edges or [],
        FETCH_EPISODES_QUERY: episodes or [],
    }
    session = SimpleNamespace(
        run=AsyncMock(
            side_effect=lambda query, **_: SimpleNamespace(
                data=AsyncMock(return_value=records[query])
            )
        )
    )
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    return SimpleNamespace(session=lambda: context), session


async def test_small_graph_includes_every_memory_and_reports_complete_metadata():
    driver, session = graph_driver(
        nodes=[{"uuid": "city", "name": "Bogota", "summary": "Home city", "degree": 2}],
        edges=[
            {
                "uuid": "lives",
                "source_name": "User",
                "target_name": "Bogota",
                "relation_name": "LIVES_IN",
                "fact": "Moved in 2020",
            }
        ],
        episodes=[
            {
                "uuid": "trip",
                "valid_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
                "summary": "Planned a hike",
            }
        ],
    )
    result = await HydrationV2Engine(driver, min_degree=1, char_limit=10000).build("owner")
    assert not result.metadata.is_partial
    assert result.metadata.included_node_ids == ["city"]
    assert result.metadata.included_edge_ids == ["lives"]
    assert result.metadata.included_episode_ids == ["trip"]
    assert "Home city" in result.compilation_text
    assert "Moved in 2020" in result.compilation_text
    assert "Planned a hike" in result.compilation_text
    for call in session.run.await_args_list:
        assert call.kwargs["group_id"] == "owner"


async def test_over_budget_graph_keeps_priority_order_and_tracks_only_included_memories():
    nodes = [
        {
            "uuid": f"node-{i}",
            "name": f"Entity {i}",
            "summary": f"unique-memory-{i} " + "x" * 90,
            "degree": 10 - i,
        }
        for i in range(10)
    ]
    driver, _ = graph_driver(nodes=nodes)
    engine = HydrationV2Engine(driver, min_degree=1, char_limit=500)
    result = await engine.build("owner")
    included = result.metadata.included_node_ids
    assert result.metadata.is_partial
    assert 0 < len(included) < len(nodes)
    assert included == [node["uuid"] for node in nodes[: len(included)]]
    for node in nodes:
        assert (node["summary"] in result.compilation_text) == (node["uuid"] in included)
    # The budget counts memory content; section headings add formatting overhead.
    assert (
        sum(
            len(f"- **{node['name']}**: {node['summary']}")
            for node in nodes
            if node["uuid"] in included
        )
        <= 500
    )
    second = await engine.build("owner")
    assert second.compilation_text == result.compilation_text
    assert second.metadata == result.metadata


async def test_new_user_with_empty_graph_returns_empty_memory():
    driver, _ = graph_driver()
    result = await HydrationV2Engine(driver, min_degree=1).build("new-user")
    assert result.compilation_text == ""
    assert result.metadata is None
