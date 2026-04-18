"""
Cache Manager - Gemini explicit context caching lifecycle.

Stateless: the cache name returned on creation is persisted client-side
and forwarded in chat requests, so the backend does not track user → cache
mappings in memory.
"""

import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Gemini Flash requires 1,024+ tokens to create a cache; ~4 chars per token
# is a conservative estimate. Below this we skip caching.
MIN_CHARS_FOR_CACHE = 4_000


class CacheManager:
    """Creates and cleans up Gemini CachedContent resources."""

    def __init__(
        self,
        raw_client: genai.Client,
        model: str,
        default_ttl: str = "3600s",
    ):
        # Uses the raw genai.Client (not the PostHog wrapper) because the
        # PostHog AsyncClient does not expose the .caches API.
        self._client = raw_client
        self._model = model
        self._ttl = default_ttl

    async def create_compilation_cache(
        self,
        user_id: str,
        compilation_text: str,
    ) -> tuple[str | None, str]:
        """Create a Gemini cache for the user's compilation.

        Returns (cache_name, skip_reason). cache_name is None when creation
        was skipped or failed. skip_reason is "compilation_too_small",
        "creation_failed", or "" on success.
        """
        if len(compilation_text) < MIN_CHARS_FOR_CACHE:
            logger.info(
                "Skipping cache creation for user=%s: compilation too small "
                "(%d chars, min %d)",
                user_id, len(compilation_text), MIN_CHARS_FOR_CACHE,
            )
            return None, "compilation_too_small"

        try:
            cache = await self._client.aio.caches.create(
                model=self._model,
                config=types.CreateCachedContentConfig(
                    display_name=f"compilation_{user_id}",
                    system_instruction=compilation_text,
                    ttl=self._ttl,
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to create Gemini cache for user=%s (model=%s, chars=%d, ttl=%s): "
                "%s: %s",
                user_id, self._model, len(compilation_text), self._ttl,
                type(e).__name__, e,
            )
            return None, "creation_failed"

        logger.info(
            "Created Gemini cache for user=%s: name=%s, chars=%d, ttl=%s",
            user_id, cache.name, len(compilation_text), self._ttl,
        )
        return cache.name, ""

    async def invalidate_by_name(self, cache_name: str) -> None:
        """Best-effort delete of a cache by its resource name."""
        try:
            await self._client.aio.caches.delete(name=cache_name)
            logger.info("Deleted Gemini cache %s", cache_name)
        except Exception as e:
            logger.debug(
                "Failed to delete cache %s (likely already gone): %s",
                cache_name, e,
            )

    async def refresh_ttl(self, cache_name: str) -> None:
        """Best-effort TTL refresh so active users don't hit expiration.

        Gemini caches expire by wallclock regardless of usage. Call this
        after a successful cache hit to push the expiration window forward.
        """
        try:
            await self._client.aio.caches.update(
                name=cache_name,
                config=types.UpdateCachedContentConfig(ttl=self._ttl),
            )
            logger.debug(
                "Refreshed TTL for cache %s to %s", cache_name, self._ttl,
            )
        except Exception as e:
            logger.debug(
                "Failed to refresh TTL for cache %s: %s", cache_name, e,
            )
