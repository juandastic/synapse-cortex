import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.services.generation import (
    GenerationService,
    _collect_grounding_metadata,
)


class _FakeModels:
    def __init__(self):
        self.kwargs = None

    async def generate_content_stream(self, **kwargs):
        self.kwargs = kwargs
        return object()


class _FakeClient:
    def __init__(self):
        self.aio = type("Aio", (), {"models": _FakeModels()})()


class GroundingConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def test_grounding_is_enabled_by_default_and_parses_false(self):
        self.assertIs(Settings.model_fields["grounding_enabled"].default, True)
        settings = Settings(
            _env_file=None,
            neo4j_password="test",
            synapse_api_secret="test",
            grounding_enabled="false",
        )
        self.assertIs(settings.grounding_enabled, False)

    async def test_google_search_tool_is_added_when_enabled(self):
        client = _FakeClient()
        service = GenerationService(client, grounding_enabled=True)

        await service._stream_gemini("model", [], None)

        config = client.aio.models.kwargs["config"]
        self.assertIsNotNone(config.tools[0].google_search)

    async def test_cached_request_does_not_redeclare_google_search(self):
        client = _FakeClient()
        service = GenerationService(client, grounding_enabled=True)

        await service._stream_gemini(
            "model",
            [],
            None,
            cache_name="cachedContents/test",
        )

        config = client.aio.models.kwargs["config"]
        self.assertEqual(config.cached_content, "cachedContents/test")
        self.assertIsNone(config.tools)

    async def test_google_search_tool_is_omitted_when_disabled(self):
        client = _FakeClient()
        service = GenerationService(client, grounding_enabled=False)

        await service._stream_gemini("model", [], None)

        self.assertNotIn("config", client.aio.models.kwargs)


class GroundingMetadataTests(unittest.TestCase):
    def test_collects_unique_queries_sources_and_supports(self):
        chunk = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        web_search_queries=["current guidance", "current guidance"],
                        search_entry_point=SimpleNamespace(
                            rendered_content="  <div>Google Search</div>  "
                        ),
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    title="Official source",
                                    uri="https://example.org/source",
                                )
                            ),
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    title="Duplicate",
                                    uri="https://example.org/source",
                                )
                            ),
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    title="Unsafe",
                                    uri="javascript:alert(1)",
                                )
                            ),
                        ],
                        grounding_supports=[
                            SimpleNamespace(
                                segment=SimpleNamespace(start_index=0, end_index=12),
                                grounding_chunk_indices=[0],
                            ),
                            SimpleNamespace(
                                segment=SimpleNamespace(start_index=13, end_index=25),
                                grounding_chunk_indices=[0],
                            ),
                        ],
                    )
                )
            ]
        )
        queries = set()
        sources = {}
        supports = set()

        rendered_search_entry_point = _collect_grounding_metadata(
            chunk,
            queries,
            sources,
            supports,
        )

        self.assertEqual(queries, {"current guidance"})
        self.assertEqual(list(sources), ["https://example.org/source"])
        self.assertEqual(sources["https://example.org/source"].title, "Official source")
        self.assertEqual(len(supports), 2)
        self.assertEqual(
            rendered_search_entry_point,
            "  <div>Google Search</div>  ",
        )


if __name__ == "__main__":
    unittest.main()
