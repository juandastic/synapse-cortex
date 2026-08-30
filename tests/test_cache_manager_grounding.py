import unittest

from app.services.cache_manager import CacheManager, MIN_CHARS_FOR_CACHE


class _FakeCaches:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Cache", (), {"name": "cachedContents/test"})()


class _FakeClient:
    def __init__(self):
        self.aio = type("Aio", (), {"caches": _FakeCaches()})()


class CacheManagerGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_contains_google_search_when_enabled(self):
        client = _FakeClient()
        manager = CacheManager(client, "model", grounding_enabled=True)

        cache_name, reason = await manager.create_compilation_cache(
            "user",
            "x" * MIN_CHARS_FOR_CACHE,
        )

        self.assertEqual(cache_name, "cachedContents/test")
        self.assertEqual(reason, "")
        config = client.aio.caches.kwargs["config"]
        self.assertIsNotNone(config.tools[0].google_search)

    async def test_cache_omits_google_search_when_disabled(self):
        client = _FakeClient()
        manager = CacheManager(client, "model", grounding_enabled=False)

        await manager.create_compilation_cache(
            "user",
            "x" * MIN_CHARS_FOR_CACHE,
        )

        config = client.aio.caches.kwargs["config"]
        self.assertIsNone(config.tools)


if __name__ == "__main__":
    unittest.main()
