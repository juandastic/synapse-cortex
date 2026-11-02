"""Build outside HTTP requests; recover queued work after process restarts."""

import asyncio
import logging
import time
from dataclasses import asdict

from app.core.posthog import posthog_user_context
from app.services.hydration_compiler import (
    COMPILER_VERSION,
    EXCLUDED_SECTION,
    HydrationCompiler,
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
            candidates = self.compiler.fact_candidates(
                [asdict(source) for source in sources],
                edges,
                episode_ids={episode["uuid"] for episode in episodes},
            )
            grouped, seen_hashes = await self._group_incremental(job, candidates)
        result, retained = await self._render_guarded(user, sources, candidates, grouped, edges)
        return await self._publish_if_unchanged(
            job,
            stamp,
            result,
            retained,
            episodes,
            edges,
            seen_hashes,
            {c["id"]: c["fact_section"] for c in grouped if not c.get("classification_fallback")},
        )

    async def _group_incremental(self, job, candidates) -> tuple[list[dict], dict[str, str]]:
        previous = await self.store.current(job["user"]) or {}
        if previous.get("compiler") != COMPILER_VERSION:
            previous = {}
        seen_hashes = {c["id"]: digest(c) for c in candidates}
        current_ids = {c["id"] for c in candidates}
        sections = {
            cid: title
            for cid, title in previous.get("fact_sections", {}).items()
            if cid in current_ids
        }
        pending = []
        grouped = {}
        for candidate in candidates:
            cid = candidate["id"]
            if previous.get("seen_hashes", {}).get(cid) == seen_hashes[cid] and sections.get(cid):
                grouped[cid] = {
                    **candidate,
                    "fact_section": sections[cid],
                    "excluded": sections[cid] == EXCLUDED_SECTION
                    and bool(candidate.get("excluded_context"))
                    and not candidate.get("correction"),
                }
            else:
                pending.append(
                    {
                        **candidate,
                        "classification_retry": previous.get("seen_hashes", {}).get(cid)
                        == seen_hashes[cid],
                    }
                )
        if pending:
            classified = await self.compiler.group(
                pending,
                cache=self.store.group_cache(job),
                existing_groups=list(dict.fromkeys(sections.values())),
                retry_epoch=job["generation"] if previous else None,
            )
            grouped.update({c["id"]: c for c in classified})
        return [grouped[c["id"]] for c in candidates], seen_hashes

    async def _render_guarded(self, user, sources, candidates, grouped, edges):
        if candidates and not grouped:
            raise ValueError("Grouping omitted every available fact; preserving legacy hydration")
        stats = await self.hydration_service.fetch_graph_stats(user)
        all_sources_excluded = (
            not grouped and bool(sources) and all(source.excluded for source in sources)
        )
        if (
            not grouped
            and not all_sources_excluded
            and (sources or stats.entity_count or stats.relationship_count)
        ):
            raise UnsupportedArchive(
                "Graph knowledge has no usable episodic evidence; preserving v2"
            )
        result, retained = self.compiler.render(grouped, edges, stats)
        if any(not c.get("excluded") for c in grouped) and not result.compilation_text:
            raise ValueError("Compilation budget omitted every available fact")
        return result, retained

    async def _publish_if_unchanged(
        self, job, stamp, result, retained, episodes, edges, seen_hashes, fact_sections
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
            job,
            stamp,
            result,
            retained,
            episodes,
            edges,
            seen_hashes=seen_hashes,
            fact_sections=fact_sections,
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
