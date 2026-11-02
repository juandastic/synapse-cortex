"""Build outside HTTP requests; recover queued work after process restarts."""

import asyncio
import logging
import time

from app.core.observability import anonymize_id
from app.core.posthog import posthog_user_context
from app.services.hydration_compiler import (
    BATCH_CHARS,
    COMPILER_VERSION,
    HydrationCompiler,
    UnverifiableEvidence,
    WorkPending,
    digest,
    sources_from_episodes,
)

logger = logging.getLogger(__name__)


class UnsupportedArchive(ValueError):
    pass


class PersistedHydration:
    def __init__(self, store, compiler: HydrationCompiler, hydration_service, cache_manager=None):
        self.store = store
        self.compiler = compiler
        self.hydration_service = hydration_service
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
        stamp = await self.store.stamp(user)
        episodes, edges = await self.store.snapshot(user)
        sources = sources_from_episodes(episodes)
        with posthog_user_context(user):
            quotes = await self._cached_quotes_or_extract(job, sources)
            candidates = self.compiler.fact_candidates(quotes, edges)
            selected, seen_hashes = await self._select_incremental(job, candidates)
        result, retained = await self._render_guarded(user, sources, candidates, selected, edges)
        return await self._publish_if_unchanged(
            job, stamp, result, retained, episodes, edges, seen_hashes
        )

    async def _cached_quotes_or_extract(self, job, sources) -> list[dict]:
        await self.store.prune_sources(job, [s.id for s in sources])
        cache = await self.store.cached_sources(job["user"])
        quotes, pending = [], []
        for source in sources:
            if source.id in cache and cache[source.id][0] == source.hash:
                quotes.extend(cache[source.id][1])
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
        logger.info(
            "Hydration user=%s: %d changed batches, %d cached quotes",
            anonymize_id(job["user"]),
            len(batches),
            len(quotes),
        )
        if not batches:
            return quotes

        async def extract_batch(batch):
            try:
                items = await self.compiler.extract(batch)
            except UnverifiableEvidence:
                logger.warning("Quarantining unverifiable hydration extraction batch")
                await self.store.save_sources(job, batch, [], quarantine_reason="nonliteral_quotes")
                return
            await self.store.save_sources(job, batch, items)

        # Persist at most two batches in parallel, then yield to other users.
        async with asyncio.TaskGroup() as group:
            for batch in batches[:2]:
                group.create_task(extract_batch(batch))
        raise WorkPending()

    async def _select_incremental(self, job, candidates) -> tuple[list[dict], dict[str, str]]:
        previous = await self.store.current(job["user"]) or {}
        if previous.get("compiler") != COMPILER_VERSION:
            previous = {}

        # Hash source facts before restoring their previous display categories.
        seen_hashes = {candidate["id"]: digest(candidate) for candidate in candidates}
        by_id = {candidate["id"]: candidate for candidate in candidates}
        previous_ids = []
        for item in previous.get("selected", []):
            if item["id"] in by_id:
                previous_ids.append(item["id"])
                by_id[item["id"]]["fact_section"] = item.get("fact_section")
        changed = [
            candidate
            for candidate in candidates
            if previous.get("seen_hashes", {}).get(candidate["id"]) != seen_hashes[candidate["id"]]
        ]
        if previous_ids and not changed:
            selected = [by_id[cid] for cid in previous_ids]
        else:
            wanted = set(previous_ids) | {candidate["id"] for candidate in changed}
            pool = [candidate for candidate in candidates if candidate["id"] in wanted]
            # If all resident facts disappeared, refill from the remaining archive.
            selected = await self.compiler.select(
                pool or candidates, cache=self.store.selection_cache(job)
            )
        corrections = [candidate for candidate in changed if candidate.get("correction")]
        selected = list(
            {candidate["id"]: candidate for candidate in corrections + selected}.values()
        )[:180]
        return selected, seen_hashes

    async def _render_guarded(self, user, sources, candidates, selected, edges):
        # Never replace a populated user's memory with an empty model response.
        if candidates and not selected:
            raise ValueError("No verifiable personal memory selected; keeping legacy hydration")
        stats = await self.hydration_service.fetch_graph_stats(user)
        if not selected and (sources or stats.entity_count or stats.relationship_count):
            raise UnsupportedArchive(
                "Graph knowledge has no usable episodic evidence; preserving v2"
            )
        result, retained = self.compiler.render(selected, edges, stats)
        if selected and not result.compilation_text:
            raise ValueError("Compilation budget omitted every selected claim")
        return result, retained

    async def _publish_if_unchanged(
        self, job, stamp, result, retained, episodes, edges, seen_hashes
    ) -> bool:
        if await self.store.stamp(job["user"]) != stamp:
            # Keep this job dirty. An old builder must not revoke a newer
            # publication if its lease was already reclaimed by another process.
            return False
        if self.cache_manager and result.compilation_text:
            try:
                name, _ = await asyncio.wait_for(
                    self.cache_manager.create_compilation_cache(
                        job["user"], result.compilation_text
                    ),
                    timeout=10,
                )
                result.provider_cache_name = name
                result.provider_cache_model = self.cache_manager.model_name
                result.provider_cache_expires_at = (
                    time.time() + self.cache_manager.ttl_seconds if name else 0
                )
            except Exception:
                logger.warning("Provider cache preparation failed; publication can be sent inline")
        if await self.store.stamp(job["user"]) != stamp:
            return False
        return await self.store.publish(
            job, stamp, result, retained, episodes, edges, seen_hashes=seen_hashes
        )

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
        except WorkPending:
            pass
        except UnsupportedArchive:
            terminal = True
            logger.info(
                "No supported profile sources; preserving legacy hydration until memory changes"
            )
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
