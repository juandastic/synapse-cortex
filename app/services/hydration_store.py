"""Neo4j materialization, source revision checks and leased durable work queue."""

import json
import logging
from dataclasses import asdict
from uuid import uuid4

from app.services.hydration_compiler import COMPILER_VERSION, WorkPending, digest
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult

logger = logging.getLogger(__name__)

STAMP_QUERY = """
 MATCH (e:Episodic {group_id:$user})
 WITH count(e) AS episodes, max(toString(e.created_at)) AS newest_episode,
 sum(size(coalesce(e.content,''))) AS episode_chars,
 sum(size(coalesce(e.summary,''))) AS summary_chars
 OPTIONAL MATCH (:Entity {group_id:$user})-[r:RELATES_TO {group_id:$user}]->(:Entity {group_id:$user})
 RETURN episodes,newest_episode,episode_chars,summary_chars,
 count(r) AS edges, count(r.expired_at) AS expired, count(r.invalid_at) AS invalid,
 sum(CASE WHEN r IS NOT NULL AND r.expired_at IS NULL AND (r.invalid_at IS NULL OR r.invalid_at>datetime())
          THEN 1 ELSE 0 END) AS active,
 max(toString(r.created_at)) AS newest_edge,
 max(toString(r.expired_at)) AS newest_expiry,
 max(toString(r.invalid_at)) AS newest_invalidation,
 sum(size(coalesce(r.fact,''))) AS fact_chars
"""
EPISODES_QUERY = """
MATCH (e:Episodic {group_id:$user})
RETURN e.uuid AS uuid, e.source AS source, e.content AS content,e.name AS name,
       toString(e.valid_at) AS valid_at
ORDER BY e.valid_at
"""
EDGES_QUERY = """
MATCH (a:Entity {group_id:$user})-[r:RELATES_TO {group_id:$user}]->(b:Entity {group_id:$user})
WHERE r.expired_at IS NULL AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
RETURN r.uuid AS uuid, r.fact AS fact, r.episodes AS episodes,
       toString(r.valid_at) AS valid_at,toString(r.invalid_at) AS invalid_at,
       a.uuid AS source_uuid, b.uuid AS target_uuid
"""

DELETE_HYDRATION_QUERY = """
MATCH (n {user_id:$user})
WHERE n:SynapseUserCore OR n:SynapseHydrationSource OR n:SynapseHydrationSelection
   OR n:SynapseHydrationRevision
DETACH DELETE n
"""


class HydrationStore:
    def __init__(self, driver):
        self.driver = driver

    async def rows(self, query, **params):
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    async def initialize(self):
        await self.rows("""CREATE CONSTRAINT synapse_user_core_user IF NOT EXISTS
            FOR (n:SynapseUserCore) REQUIRE n.user_id IS UNIQUE""")
        await self.rows("""CREATE CONSTRAINT synapse_hydration_source_key IF NOT EXISTS
            FOR (n:SynapseHydrationSource) REQUIRE n.key IS UNIQUE""")
        await self.rows("""CREATE INDEX synapse_hydration_source_user IF NOT EXISTS
            FOR (n:SynapseHydrationSource) ON (n.user_id)""")
        await self.rows("""CREATE CONSTRAINT synapse_hydration_selection_key IF NOT EXISTS
            FOR (n:SynapseHydrationSelection) REQUIRE n.key IS UNIQUE""")
        await self.rows("""CREATE INDEX synapse_hydration_selection_user IF NOT EXISTS
            FOR (n:SynapseHydrationSelection) ON (n.user_id)""")

    async def stamp(self, user):
        return digest(await self.rows(STAMP_QUERY, user=user))

    async def request(self, user, *, invalidate=False, source_changed=False):
        await self.rows(
            """
        MERGE (n:SynapseUserCore {user_id:$user})
        ON CREATE SET n.hydration_generation=0
        SET n.hydration_lock = coalesce(n.hydration_lock,0)+1
        WITH n, $invalidate OR $source_changed AS fresh
        SET n.hydration_generation = CASE WHEN fresh
            THEN coalesce(n.hydration_generation,0)+1 ELSE n.hydration_generation END,
            n.hydration_dirty = true,
            n.hydration_retry_at = CASE WHEN fresh THEN 0 ELSE coalesce(n.hydration_retry_at,0) END,
            n.hydration_attempts=CASE WHEN fresh THEN 0 ELSE coalesce(n.hydration_attempts,0) END,
            n.hydration_failed_generation=CASE WHEN fresh THEN null ELSE n.hydration_failed_generation END,
            n.hydration_payload = CASE WHEN $invalidate THEN null ELSE n.hydration_payload END
        """,
            user=user,
            invalidate=invalidate,
            source_changed=source_changed,
        )

    async def discard(self, user, revision):
        # Reads may revoke this payload, but must never cancel a newer builder
        # or remove a snapshot that was published after the read began.
        await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user})
            SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
            WITH n WHERE n.hydration_revision=$revision
            SET n.hydration_payload=null,n.hydration_dirty=true""",
            user=user,
            revision=revision,
        )

    @staticmethod
    def result(payload):
        data = payload["result"]
        return HydrationResult(
            **{
                **data,
                "metadata": CompilationMetadata(**data["metadata"]),
                "graph_stats": GraphStats(**data["graph_stats"]),
            }
        )

    async def chat_read(self, user, revision):
        rows = await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user})
            RETURN n.hydration_revision AS revision,n.hydration_payload IS NOT NULL AS available,
            CASE WHEN $revision IS NULL OR n.hydration_revision<>$revision
                THEN n.hydration_payload ELSE null END AS payload,
            n.hydration_cache_name AS cache_name,n.hydration_cache_model AS cache_model,
            n.hydration_cache_expires_at AS cache_expires
            """,
            user=user,
            revision=revision,
        )
        if not rows:
            return "absent", None
        if not rows[0]["available"]:
            return "revoked", None
        row = rows[0]
        if row["revision"] == revision:
            return "unchanged", None
        payload = json.loads(row["payload"])
        if payload["compiler"] != COMPILER_VERSION:
            return "unchanged", None
        result = self.result(payload)
        result.provider_cache_name = row.get("cache_name")
        result.provider_cache_model = row.get("cache_model")
        result.provider_cache_expires_at = row.get("cache_expires") or 0
        return "updated", result

    async def current(self, user):
        rows = await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user})
            RETURN n.hydration_payload AS payload,n.hydration_cache_name AS cache_name,
                n.hydration_cache_model AS cache_model,n.hydration_cache_expires_at AS cache_expires""",
            user=user,
        )
        if not rows or not rows[0]["payload"]:
            return None
        payload = json.loads(rows[0]["payload"])
        payload["result"].update(
            provider_cache_name=rows[0].get("cache_name"),
            provider_cache_model=rows[0].get("cache_model"),
            provider_cache_expires_at=rows[0].get("cache_expires") or 0,
        )
        return payload

    async def set_cache(self, user, revision, name, model, expires):
        await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_revision:$revision})
            WHERE n.hydration_payload IS NOT NULL
            SET n.hydration_cache_name=$name,n.hydration_cache_model=$model,
                n.hydration_cache_expires_at=$expires""",
            user=user,
            revision=revision,
            name=name,
            model=model,
            expires=expires,
        )

    async def read(self, user):
        payload = await self.current(user)
        if payload is None or payload["compiler"] != COMPILER_VERSION:
            return None
        stamp_changed = payload["stamp"] != await self.stamp(user)
        # Detect same-length edits or removal of selected evidence as well.
        rows = await self.rows(
            """MATCH (e:Episodic {group_id:$user}) WHERE e.uuid IN $ids
            RETURN e.uuid AS uuid, e.content AS content, e.source AS source,
                   toString(e.valid_at) AS valid_at,e.name AS name""",
            user=user,
            ids=list(payload["source_hashes"]),
        )
        actual = {r["uuid"]: digest(r) for r in rows}
        if actual != payload["source_hashes"]:
            await self.discard(user, payload["result"]["metadata"]["compilation_revision"])
            return None
        covered = payload.get("edge_hashes", {})
        if covered:
            rows = await self.rows(
                """MATCH (:Entity {group_id:$user})
                -[r:RELATES_TO {group_id:$user}]->(:Entity {group_id:$user})
                WHERE r.uuid IN $ids AND r.expired_at IS NULL
                  AND (r.invalid_at IS NULL OR r.invalid_at>datetime())
                RETURN r.uuid AS uuid,r.fact AS fact,
                    toString(r.valid_at) AS valid_at,toString(r.invalid_at) AS invalid_at""",
                user=user,
                ids=list(covered),
            )
            if {
                r["uuid"]: digest([r["fact"], r["valid_at"], r["invalid_at"]]) for r in rows
            } != covered:
                await self.discard(user, payload["result"]["metadata"]["compilation_revision"])
                return None
        if stamp_changed:
            # New evidence is additive when all published dependencies still
            # match. Serve the prepared version and enqueue its successor.
            await self.request(user)
        return self.result(payload)

    async def claim(self, user=None):
        token = str(uuid4())
        rows = await self.rows(
            """
        MATCH (n:SynapseUserCore)
        WHERE ($user IS NULL OR n.user_id=$user)
          AND n.hydration_dirty = true AND coalesce(n.hydration_retry_at,0) <= timestamp()
          AND (n.hydration_failed_generation IS NULL OR n.hydration_failed_generation<>n.hydration_generation)
          AND coalesce(n.hydration_lease_until,0) < timestamp()
        WITH n ORDER BY coalesce(n.hydration_retry_at,0) LIMIT 1
        SET n.hydration_lock = coalesce(n.hydration_lock,0)+1
        WITH n WHERE n.hydration_dirty = true
          AND coalesce(n.hydration_lease_until,0) < timestamp()
        SET n.hydration_owner=$token, n.hydration_lease_until=timestamp()+120000
        RETURN n.user_id AS user, n.hydration_generation AS generation,
               coalesce(n.hydration_attempts,0) AS attempts
        """,
            token=token,
            user=user,
        )
        return {**rows[0], "token": token} if rows else None

    async def renew(self, job):
        rows = await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
            WHERE n.hydration_generation=$generation AND n.hydration_lease_until>timestamp()
            SET n.hydration_lease_until=timestamp()+120000 RETURN n.user_id AS owner""",
            **job,
        )
        return bool(rows)

    async def release(self, job, *, failed=False, terminal=False):
        attempts = job.get("attempts", 0) + int(failed)
        delay = min(900000, 30000 * (2 ** min(attempts - 1, 5))) if failed else 1000
        await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
            SET n.hydration_lease_until=0, n.hydration_owner=null,
                n.hydration_retry_at=CASE WHEN n.hydration_generation=$generation
                    THEN timestamp()+$delay ELSE n.hydration_retry_at END,
                n.hydration_attempts=CASE WHEN n.hydration_generation=$generation
                    THEN $next_attempts ELSE n.hydration_attempts END,
                n.hydration_failed_generation=CASE
                    WHEN n.hydration_generation=$generation AND ($terminal OR $next_attempts>=5)
                    THEN $generation ELSE n.hydration_failed_generation END
            """,
            **job,
            delay=delay,
            next_attempts=attempts,
            terminal=terminal,
        )

    async def snapshot(self, user):
        episodes = await self.rows(EPISODES_QUERY, user=user)
        edges = await self.rows(EDGES_QUERY, user=user)
        return episodes, edges

    async def cached_sources(self, user):
        rows = await self.rows(
            """MATCH (s:SynapseHydrationSource {user_id:$user})
            RETURN s.source_id AS source_id,s.source_hash AS hash,s.items AS items""",
            user=user,
        )
        return {r["source_id"]: (r["hash"], json.loads(r["items"])) for r in rows}

    async def save_sources(self, job, sources, items):
        user = job["user"]
        records = [
            {
                "key": digest([user, s.id]),
                "id": s.id,
                "hash": s.hash,
                "items": json.dumps(
                    [c for c in items if c["source_id"] == s.id], ensure_ascii=False
                ),
            }
            for s in sources
        ]
        rows = await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
            SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
            WITH n WHERE n.hydration_generation=$generation AND n.hydration_lease_until>timestamp()
            UNWIND $records AS r
            MERGE (s:SynapseHydrationSource {key:r.key})
            SET s.user_id=$user,s.source_id=r.id,s.source_hash=r.hash,s.items=r.items
            RETURN count(s) AS saved
            """,
            **job,
            records=records,
        )
        if not rows or rows[0]["saved"] != len(records):
            raise RuntimeError("Hydration source writer lost its lease or revision")

    async def prune_sources(self, job, source_ids):
        rows = await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
            SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
            WITH n WHERE n.hydration_generation=$generation AND n.hydration_lease_until>timestamp()
            OPTIONAL MATCH (s:SynapseHydrationSource {user_id:$user})
            WHERE NOT s.source_id IN $ids
            WITH n,collect(s) AS obsolete
            FOREACH (s IN obsolete | DELETE s)
            RETURN n.user_id AS owner""",
            **job,
            ids=source_ids,
        )
        if not rows:
            raise RuntimeError("Hydration source cleanup lost its lease or revision")

    async def publish(self, job, stamp, result, selected, episodes, edges):
        revision = str(uuid4())
        result.metadata.compilation_revision = revision
        source_ids = set(result.metadata.included_episode_ids)
        hashes = {e["uuid"]: digest(e) for e in episodes if e["uuid"] in source_ids}
        payload = json.dumps(
            {
                "compiler": COMPILER_VERSION,
                "stamp": stamp,
                "result": asdict(result),
                "source_hashes": hashes,
                "selected": selected,
                "seen_hashes": getattr(result, "seen_hashes", {}),
                "edge_hashes": {
                    e["uuid"]: digest([e["fact"], e.get("valid_at"), e.get("invalid_at")])
                    for e in edges
                    if e["uuid"] in result.metadata.included_edge_ids
                },
            },
            ensure_ascii=False,
        )
        # Audit revisions keep references and hashes, never historic copies of quotations.
        rows = await self.rows(
            """
        MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
        SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
        WITH n WHERE n.hydration_generation=$generation
          AND n.hydration_lease_until>timestamp()
        SET n.hydration_payload=$payload,n.hydration_dirty=false,
            n.hydration_revision=$revision,n.hydration_updated_at=datetime(),
            n.hydration_attempts=0,n.hydration_failed_generation=null,
            n.hydration_cache_name=$cache_name,n.hydration_cache_model=$cache_model,
            n.hydration_cache_expires_at=$cache_expires
        CREATE (:SynapseHydrationRevision {user_id:$user, revision:$revision,
            source_stamp:$stamp,claim_ids:$claim_ids,content_hash:$hash,created_at:datetime()})
        RETURN n.hydration_revision AS revision
        """,
            **job,
            revision=revision,
            payload=payload,
            stamp=stamp,
            claim_ids=[c["id"] for c in selected],
            hash=digest(result.compilation_text),
            cache_name=result.provider_cache_name,
            cache_model=result.provider_cache_model,
            cache_expires=result.provider_cache_expires_at,
        )
        if rows:
            try:
                await self.rows(
                    """MATCH (r:SynapseHydrationRevision {user_id:$user})
                    WITH r ORDER BY r.created_at DESC SKIP 20 DELETE r""",
                    user=job["user"],
                )
                await self.rows(
                    """MATCH (r:SynapseHydrationSelection {user_id:$user})
                    WITH r ORDER BY r.created_at DESC SKIP 100 DELETE r""",
                    user=job["user"],
                )
            except Exception:
                logger.warning(
                    "Hydration retention cleanup failed after publication", exc_info=True
                )
        return bool(rows)

    def selection_cache(self, job):
        return SelectionCache(self, job)


class SelectionCache:
    """Cache only chosen IDs; never duplicate personal text in historical caches."""

    def __init__(self, store, job):
        self.store, self.job = store, job
        self.calls = 0

    async def load(self, fingerprint):
        rows = await self.store.rows(
            """MATCH (n:SynapseHydrationSelection {key:$key})
            RETURN n.ids AS ids""",
            key=digest([self.job["user"], fingerprint]),
        )
        if rows:
            return rows[0]["ids"]
        if self.calls >= 2:
            raise WorkPending()
        self.calls += 1
        return None

    async def save(self, fingerprint, ids):
        rows = await self.store.rows(
            """MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
            SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
            WITH n WHERE n.hydration_generation=$generation AND n.hydration_lease_until>timestamp()
            MERGE (s:SynapseHydrationSelection {key:$key})
            SET s.user_id=$user,s.ids=$ids,s.created_at=datetime()
            RETURN s.key AS key""",
            **self.job,
            key=digest([self.job["user"], fingerprint]),
            ids=ids,
        )
        if not rows:
            raise RuntimeError("Hydration selector lost its lease or revision")
