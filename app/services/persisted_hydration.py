"""Build outside HTTP requests; recover queued work after process restarts."""

import asyncio
import logging

from app.core.observability import anonymize_id
from app.core.posthog import posthog_user_context
from app.services.hydration_compiler import BATCH_CHARS, HydrationCompiler, sources_from_episodes

logger = logging.getLogger(__name__)


class PersistedHydration:
    def __init__(self, store, compiler: HydrationCompiler, legacy):
        self.store = store
        self.compiler = compiler
        self.legacy = legacy
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
        await self.store.request(user, invalidate=invalidate)
        self._wake.set()

    async def read(self, user, *, strict=False):
        try:
            result = await self.store.read(user)
            if result is None:
                await self.changed(user)
            return result
        except Exception:
            if strict:
                raise
            logger.exception("Persisted hydration unavailable; using legacy hydration")
            return None

    async def _heartbeat(self, job):
        while True:
            await asyncio.sleep(30)
            await self.store.renew(job)

    async def build(self, job):
        user = job["user"]
        stamp = await self.store.stamp(user)
        episodes, edges = await self.store.snapshot(user)
        sources = sources_from_episodes(episodes)
        await self.store.prune_sources(job, [s.id for s in sources])
        cache = await self.store.cached_sources(user)
        candidates, pending = [], []
        for source in sources:
            if source.id in cache and cache[source.id][0] == source.hash:
                candidates.extend(cache[source.id][1])
            else:
                pending.append(source)
        batches, batch, size = [], [], 0
        for source in pending:
            if batch and size + len(source.text) > BATCH_CHARS:
                batches.append(batch)
                batch, size = [], 0
            batch.append(source)
            size += len(source.text)
        if batch:
            batches.append(batch)
        with posthog_user_context(user):
            semaphore = asyncio.Semaphore(2)

            async def extract_batch(batch):
                async with semaphore:
                    items = await asyncio.wait_for(self.compiler.extract(batch), timeout=120)
                    await self.store.save_sources(job, batch, items)
                    return items

            logger.info(
                "Hydration user=%s: %d changed batches, %d cached candidates",
                anonymize_id(user),
                len(batches),
                len(candidates),
            )
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(extract_batch(batch)) for batch in batches]
            for task in tasks:
                candidates.extend(task.result())
            selected = await self.compiler.select(candidates)
        # Never replace a populated user's memory with an empty model response.
        if candidates and not selected:
            raise ValueError("No verifiable personal memory selected; keeping legacy hydration")
        if sources and not candidates:
            previous = await self.store.current(user)
            if previous and previous["result"]["compilation_text"]:
                raise ValueError("Empty extraction cannot replace an existing personal profile")
        stats = await self.legacy._fetch_graph_stats(user)
        if not selected and (stats.entity_count or stats.relationship_count):
            raise ValueError("Graph knowledge has no usable episodic evidence; preserving v2")
        result, retained = self.compiler.render(selected, edges, stats)
        if selected and not result.compilation_text:
            raise ValueError("Compilation budget omitted every selected claim")
        if await self.store.stamp(user) != stamp:
            # Keep this job dirty. An old builder must not revoke a newer
            # publication if its lease was already reclaimed by another process.
            return False
        return await self.store.publish(job, stamp, result, retained, episodes, edges)

    async def run_one(self, user=None):
        job = await self.store.claim(user=user) if user else await self.store.claim()
        if not job:
            return False
        heartbeat = asyncio.create_task(self._heartbeat(job))
        failed = False
        try:
            await asyncio.wait_for(self.build(job), timeout=3600)
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
            logger.exception("Hydration compilation failed; work retained for retry")
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            await self.store.release(job, failed=failed)
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
