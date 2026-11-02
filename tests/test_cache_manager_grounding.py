import asyncio
import unittest
from unittest.mock import AsyncMock

from app.services.cache_manager import MIN_CHARS_FOR_CACHE, CacheManager


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
    async def test_identical_hydration_reuses_cache_until_expiry_or_invalidation(self):
        client = _FakeClient()
        client.aio.caches.create = AsyncMock(
            return_value=type("Cache", (), {"name": "cachedContents/test"})()
        )
        client.aio.caches.delete = AsyncMock()
        manager = CacheManager(client, "model", grounding_enabled=False)
        text = "x" * MIN_CHARS_FOR_CACHE
        first, second = await asyncio.gather(
            manager.create_compilation_cache("owner", text),
            manager.create_compilation_cache("owner", text),
        )
        self.assertEqual(first, second)
        self.assertEqual(client.aio.caches.create.await_count, 1)
        await manager.invalidate_by_name(first[0])
        await manager.create_compilation_cache("owner", text)
        self.assertEqual(client.aio.caches.create.await_count, 2)
        manager._recent = {key: (name, 0) for key, (name, _) in manager._recent.items()}
        await manager.create_compilation_cache("owner", text)
        self.assertEqual(client.aio.caches.create.await_count, 3)
        await manager.create_compilation_cache("another-owner", text)
        self.assertEqual(client.aio.caches.create.await_count, 4)

    async def test_cache_contains_google_search_when_enabled(self):
        for model in ("gemini-3.1-pro-preview", "gemini-4-argon"):
            with self.subTest(model=model):
                client = _FakeClient()
                manager = CacheManager(client, model, grounding_enabled=True)
                cache_name, reason = await manager.create_compilation_cache(
                    "user", "x" * MIN_CHARS_FOR_CACHE
                )
                if model == "gemini-4-argon":
                    self.assertIsNone(cache_name)
                    self.assertEqual(reason, "implicit_cache_only")
                    self.assertIsNone(client.aio.caches.kwargs)
                else:
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
