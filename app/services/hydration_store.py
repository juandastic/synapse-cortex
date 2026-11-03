"""Neo4j materialization, source revision checks and leased durable work queue."""

import json
from dataclasses import asdict
from uuid import uuid4

from app.services.hydration_compiler import COMPILER_VERSION, fact_hash
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.hydration_selection import HEAD_QUERY, VALIDATE_QUERY, select

DELETE_HYDRATION_QUERY = """
MATCH (n {user_id:$user})
WHERE n:SynapseUserCore OR n:SynapseHydrationSource OR n:SynapseHydrationSelection
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

    async def head(self, user):
        rows = await self.rows(HEAD_QUERY, user=user)
        return rows[0]["uuid"] if rows else None

    async def select(self, user):
        return await select(self, user)

    async def validate(self, user, dependencies):
        rows = await self.rows(VALIDATE_QUERY, user=user, ids=list(dependencies))
        return {r["uuid"]: fact_hash(r) for r in rows} == dependencies

    async def request(self, user, *, invalidate=False, source_changed=False):
        await self.rows(
            """
        MERGE (n:SynapseUserCore {user_id:$user})
        ON CREATE SET n.hydration_generation=0
        SET n.hydration_lock = coalesce(n.hydration_lock,0)+1
        WITH n, $invalidate OR $source_changed OR coalesce(n.hydration_requested_compiler,'')<>$compiler AS fresh
        SET n.hydration_generation = CASE WHEN fresh
            THEN coalesce(n.hydration_generation,0)+1 ELSE coalesce(n.hydration_generation,0) END,
            n.hydration_dirty = true, n.hydration_requested_compiler=$compiler,
            n.hydration_retry_at = CASE WHEN fresh THEN 0 ELSE coalesce(n.hydration_retry_at,0) END,
            n.hydration_attempts=CASE WHEN fresh THEN 0 ELSE coalesce(n.hydration_attempts,0) END,
            n.hydration_failed_generation=CASE WHEN fresh THEN null ELSE n.hydration_failed_generation END,
            n.hydration_payload = CASE WHEN $invalidate THEN null ELSE n.hydration_payload END
        SET n.hydration_result = CASE WHEN $invalidate THEN null ELSE n.hydration_result END
        """,
            user=user,
            invalidate=invalidate,
            source_changed=source_changed,
            compiler=COMPILER_VERSION,
        )

    async def discard(self, user, revision):
        # Reads may revoke this payload, but must never cancel a newer builder
        # or remove a snapshot that was published after the read began.
        await self.rows(
            """MATCH (n:SynapseUserCore {user_id:$user})
            SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
            WITH n WHERE n.hydration_revision=$revision
            SET n.hydration_payload=null,n.hydration_result=null,n.hydration_dirty=true""",
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
            RETURN n.hydration_revision AS revision,n.hydration_result IS NOT NULL AS available,
            CASE WHEN $revision IS NULL OR n.hydration_revision<>$revision
                THEN n.hydration_result ELSE null END AS payload,
            n.hydration_compiler AS compiler,
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
        if row.get("compiler") != COMPILER_VERSION:
            return "unchanged", None
        result = self.result({"result": json.loads(row["payload"])})
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
        if not await self.validate(user, payload["deps"]):
            await self.discard(user, payload["result"]["metadata"]["compilation_revision"])
            return None
        if payload["head_episode"] != await self.head(user):
            # Recover a missed best-effort ingestion notification. Reads do not
            # advance generation or interfere with a builder already in flight.
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
                n.hydration_attempts=CASE WHEN n.hydration_dirty=false THEN 0
                    WHEN n.hydration_generation=$generation
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

    async def publish(self, job, result, *, dependencies, head):
        revision = str(uuid4())
        result.metadata.compilation_revision = revision
        payload = json.dumps(
            {
                "compiler": COMPILER_VERSION,
                "deps": dependencies,
                "head_episode": head,
                "result": asdict(result),
            },
            ensure_ascii=False,
        )
        rows = await self.rows(
            """
        MATCH (n:SynapseUserCore {user_id:$user,hydration_owner:$token})
        SET n.hydration_lock=coalesce(n.hydration_lock,0)+1
        WITH n WHERE n.hydration_generation=$generation
          AND n.hydration_lease_until>timestamp()
        SET n.hydration_payload=$payload,n.hydration_dirty=false,
            n.hydration_result=$result,n.hydration_compiler=$compiler,
            n.hydration_revision=$revision,n.hydration_updated_at=datetime(),
            n.hydration_attempts=0,n.hydration_failed_generation=null,
            n.hydration_cache_name=$cache_name,n.hydration_cache_model=$cache_model,
            n.hydration_cache_expires_at=$cache_expires
        RETURN n.hydration_revision AS revision
        """,
            **job,
            revision=revision,
            payload=payload,
            result=json.dumps(asdict(result), ensure_ascii=False),
            compiler=COMPILER_VERSION,
            cache_name=result.provider_cache_name,
            cache_model=result.provider_cache_model,
            cache_expires=result.provider_cache_expires_at,
        )
        return bool(rows)
