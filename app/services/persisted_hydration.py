"""Build outside HTTP requests; recover queued work after process restarts."""

import asyncio
import logging
import time

from app.core.posthog import posthog_user_context
from app.services.hydration_compiler import (
    COMPILER_VERSION,
    HydrationCompiler,
    fact_hash,
)
from app.services.hydration_consolidation import apply_plan, organize

logger = logging.getLogger(__name__)


class UnsupportedArchive(ValueError):
    pass


class PersistedHydration:
    def __init__(self, store, compiler: HydrationCompiler, cache_manager=None):
        self.store = store
        self.compiler = compiler
        self.cache_manager = cache_manager
        self._wake = asyncio.Event()
        self._task = None

    async def start(self):
        await self.store.initialize()
        self._task = asyncio.create_task(self._run())

    async def close(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def changed(self, user, *, invalidate=False):
        await self.store.request(user, invalidate=invalidate, source_changed=True)
        if not invalidate:
            # Ingestion can retire an old fact too. Validate dependencies once
            # at write time, outside chat, before retaining the old snapshot.
            await self.store.read(user)
        self._wake.set()

    async def read(self, user):
        try:
            result = await self.store.read(user)
            if result is None:
                await self.store.request(user)
                self._wake.set()
            return result
        except Exception:
            logger.exception("Persisted hydration unavailable; using legacy hydration")
            return None

    async def _heartbeat(self, job, build_task, lost):
        while True:
            await asyncio.sleep(30)
            try:
                if not await self.store.renew(job):
                    lost.set()
                    build_task.cancel()
                    return
            except Exception:
                logger.warning("Hydration lease renewal failed; retrying", exc_info=True)

    async def build(self, job) -> bool:
        user = job["user"]
        edges, head, stats = await self.store.select(user)
        candidates = self.compiler.fact_candidates(edges)
        if not candidates and (head or stats.entity_count or stats.relationship_count):
            raise UnsupportedArchive("No active graph facts; preserving classic hydration")
        fingerprint = self.compiler.fingerprint(candidates)
        previous = await self.store.current(user) or {}
        with posthog_user_context(user):
            if (
                previous.get("compiler") == COMPILER_VERSION
                and previous.get("fingerprint") == fingerprint
            ):
                plan, reusable = previous["llm"], True
            else:
                plan, reusable = await organize(candidates, self.compiler.llm)
        records = apply_plan(candidates, plan)
        result, _ = self.compiler.render(records, edges, stats)
        if candidates and not result.compilation_text:
            raise ValueError("Compilation budget omitted every available fact")
        dependencies = {
            e["uuid"]: fact_hash(e) for e in edges if e["uuid"] in result.metadata.included_edge_ids
        }
        await self._prepare_provider_cache(user, result)
        # No archive scan: validate only facts actually rendered, after the
        # provider-cache await and immediately before the fenced publication.
        if not await self.store.validate(user, dependencies) or await self.store.head(user) != head:
            return False
        logger.info(
            "Hydration bounded build: %d selected, %d covered, reused=%s",
            len(candidates),
            len(dependencies),
            previous.get("fingerprint") == fingerprint,
        )
        return await self.store.publish(
            job,
            result,
            fingerprint=fingerprint if reusable else None,
            plan=plan,
            dependencies=dependencies,
            head=head,
        )

    async def _prepare_provider_cache(self, user, result):
        if not self.cache_manager or not result.compilation_text:
            return
        try:
            name, _ = await asyncio.wait_for(
                self.cache_manager.create_compilation_cache(user, result.compilation_text),
                timeout=10,
            )
            result.provider_cache_name = name
            result.provider_cache_model = self.cache_manager.model_name
            result.provider_cache_expires_at = (
                time.time() + self.cache_manager.ttl_seconds if name else 0
            )
        except Exception:
            logger.warning("Provider cache preparation failed; publication can be sent inline")

    async def run_one(self, user=None):
        job = await self.store.claim(user=user) if user else await self.store.claim()
        if not job:
            return False
        lost = asyncio.Event()
        build_task = asyncio.create_task(asyncio.wait_for(self.build(job), timeout=300))
        heartbeat = asyncio.create_task(self._heartbeat(job, build_task, lost))
        failed = False
        terminal = False
        try:
            await build_task
        except UnsupportedArchive:
            terminal = True
            logger.info("No active graph facts; preserving classic hydration until memory changes")
        except asyncio.CancelledError:
            if not lost.is_set():
                raise
        except Exception:
            failed = True
            logger.exception("Hydration compilation failed; work retained for retry")
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self.store.release(job, failed=failed, terminal=terminal)
        return True

    async def _run(self):
        while True:
            try:
                if await self.run_one():
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=10)
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Hydration worker temporarily unavailable")
                await asyncio.sleep(10)
