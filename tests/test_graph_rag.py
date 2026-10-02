from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.models import ChatCompletionRequest, ChatMessage, CompilationMetadataResponse
from app.services.graph_rag import build_search_query, maybe_run_graph_rag


def chat_request(**overrides):
    return ChatCompletionRequest(
        **{
            "user_id": "owner",
            "system_instruction": "Be kind",
            "compilation": "Already compiled memories",
            "compilationMetadata": CompilationMetadataResponse(
                is_partial=True,
                total_estimated_tokens=100,
                included_node_ids=["known-node"],
                included_edge_ids=["known-edge"],
            ),
            "messages": [ChatMessage(role="user", content="What about my hobby?")],
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"user_id": None}, "no_user_id"),
        ({"compilationMetadata": None}, "no_compilation_metadata"),
        (
            {
                "compilationMetadata": CompilationMetadataResponse(
                    is_partial=False,
                    total_estimated_tokens=100,
                    included_node_ids=[],
                    included_edge_ids=[],
                )
            },
            "graph_fully_loaded",
        ),
    ],
)
async def test_skips_retrieval_when_it_cannot_add_memory(overrides, reason):
    graph = SimpleNamespace(search_=AsyncMock())
    request = chat_request(**overrides)
    original = request.model_dump()
    outcome = await maybe_run_graph_rag(request, graph)
    assert not outcome.enabled
    assert outcome.skip_reason == reason
    graph.search_.assert_not_awaited()
    assert request.model_dump() == original


async def test_search_is_scoped_to_user_and_deduplicates_compiled_memory():
    graph = SimpleNamespace(
        search_=AsyncMock(
            return_value=SimpleNamespace(
                edges=[
                    SimpleNamespace(
                        uuid="known-edge", name="LIVES_IN", fact="already known", valid_at=None
                    ),
                    SimpleNamespace(
                        uuid="new-edge", name="ENJOYS", fact="likes hiking", valid_at=None
                    ),
                ],
                nodes=[
                    SimpleNamespace(uuid="known-node", name="Old city", summary="already known"),
                    SimpleNamespace(
                        uuid="new-node", name="Mountain", summary="weekend destination"
                    ),
                ],
            )
        )
    )
    request = chat_request()
    original_messages = [message.model_dump() for message in request.messages]
    outcome = await maybe_run_graph_rag(request, graph)
    assert outcome.enabled
    assert outcome.result.injected_edges_count == 1
    assert outcome.result.injected_nodes_count == 1
    assert graph.search_.await_args.kwargs["group_ids"] == ["owner"]
    assert "Be kind" in request.system_instruction
    assert "likes hiking" in request.system_instruction
    assert "weekend destination" in request.system_instruction
    assert "already known" not in request.system_instruction
    assert request.compilation == "Already compiled memories"
    assert [message.model_dump() for message in request.messages] == original_messages


async def test_legacy_messages_receive_memory_without_mutating_original_history():
    graph = SimpleNamespace(
        search_=AsyncMock(
            return_value=SimpleNamespace(
                edges=[
                    SimpleNamespace(uuid="new", name="ENJOYS", fact="likes hiking", valid_at=None)
                ],
                nodes=[],
            )
        )
    )
    system = ChatMessage(role="system", content="Original persona")
    user = ChatMessage(role="user", content="Hello")
    request = chat_request(system_instruction=None, messages=[system, user])
    await maybe_run_graph_rag(request, graph)
    assert system.content == "Original persona"
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert "Original persona" in request.messages[0].content
    assert "likes hiking" in request.messages[0].content
    assert request.messages[1] == user


async def test_retrieval_failure_keeps_chat_available_and_preserves_request():
    graph = SimpleNamespace(search_=AsyncMock(side_effect=RuntimeError("Neo4j unavailable")))
    request = chat_request()
    original = request.model_dump()
    outcome = await maybe_run_graph_rag(request, graph)
    assert not outcome.enabled
    assert outcome.skip_reason == "error"
    assert request.model_dump() == original


def test_short_followup_search_keeps_recent_context_and_excludes_system_and_images():
    messages = [
        ChatMessage(role="user", content="Old unrelated topic"),
        ChatMessage(role="system", content="Secret system instructions"),
        ChatMessage(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}},
                {"type": "text", "text": "My trip to Bogota"},
            ],
        ),
        ChatMessage(role="assistant", content="How was the hike?"),
        ChatMessage(role="user", content="Why?"),
    ]
    query = build_search_query(messages)
    assert "Bogota" in query
    assert "hike" in query
    assert "Why?" in query
    assert "Secret" not in query
    assert "photo.png" not in query
    assert "unrelated" not in query
