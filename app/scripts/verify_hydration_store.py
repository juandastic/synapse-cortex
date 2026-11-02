"""Exercise real lease/read/backoff races on loopback Neo4j, without LLM calls."""

import argparse
import asyncio
import json
import time
from urllib.parse import urlsplit
from uuid import uuid4

from neo4j import AsyncGraphDatabase

from app.core.config import Settings
from app.services.hydration_compiler import HydrationCompiler, Source
from app.services.hydration_result import GraphStats
from app.services.hydration_store import DELETE_HYDRATION_QUERY, HydrationStore


async def main(env_file):
    settings = Settings(_env_file=env_file)
    if urlsplit(settings.neo4j_uri).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Verification requires loopback Neo4j")
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    store = HydrationStore(driver)
    user = f"hydration-store-check-{uuid4()}"
    compiler = HydrationCompiler(None)
    try:
        await store.initialize()
        await store.rows(
            "CREATE (e:Episodic {group_id:$user,uuid:'ep',source:'message',name:'session',"
            "content:'User: Mi gato se llama Leo.',valid_at:datetime(),created_at:datetime()}) "
            "CREATE (a:Entity {group_id:$user,uuid:'person'}) "
            "CREATE (b:Entity {group_id:$user,uuid:'cat'}) "
            "CREATE (a)-[:RELATES_TO {group_id:$user,uuid:'fact',fact:'El gato se llama Leo',"
            "episodes:['ep'],valid_at:datetime(),created_at:datetime()}]->(b)",
            user=user,
        )
        # Coexist with a pre-existing core that has no hydration properties.
        await store.rows("CREATE (:SynapseUserCore {user_id:$user,version:1})", user=user)
        await store.request(user)
        job = await store.claim(user)
        assert job and not await store.claim(user)
        episodes, edges = await store.snapshot(user)
        quote = dict(
            id="q",
            source_id="s",
            episode_id="ep",
            date="2026-11-01",
            attribution="usuario",
            quote="Mi gato se llama Leo.",
            section="Personas y vínculos",
            kind="relato",
        )
        selected = compiler.fact_candidates([quote], edges)
        result, retained = compiler.render(selected, edges, GraphStats(2, 1))
        seen_hashes = {f"id-{i}": "x" * 64 for i in range(1000)}
        assert await store.publish(
            job, await store.stamp(user), result, retained, episodes, edges, seen_hashes=seen_hashes
        )
        await store.release(job)
        assert (await store.read(user)).metadata.included_edge_ids == ["fact"]
        audits = await store.rows(
            "MATCH (n:SynapseHydrationRevision {user_id:$user}) RETURN count(n) AS count",
            user=user,
        )
        assert audits[0]["count"] == 0
        state, refreshed = await store.chat_read(user, "previous-revision")
        assert state == "updated" and refreshed.compilation_text == result.compilation_text
        wire = await store.rows(
            "MATCH (n:SynapseUserCore {user_id:$user}) RETURN "
            "n.hydration_result AS result,n.hydration_payload AS payload",
            user=user,
        )
        assert "seen_hashes" not in json.loads(wire[0]["result"])
        assert len(wire[0]["payload"]) > 20 * len(wire[0]["result"])

        # A new episode does not revoke the old prepared snapshot or cancel its builder.
        await store.rows(
            "CREATE (:Episodic {group_id:$user,uuid:'new',source:'message',"
            "content:'User: Nuevo proyecto.',valid_at:datetime(),created_at:datetime()})",
            user=user,
        )
        await store.request(user, source_changed=True)
        job = await store.claim(user)
        assert job
        assert await store.read(user)
        assert await store.renew(job)
        source = Source("s", "ep", "2026-11-01", "usuario", "Mi gato se llama Leo.")
        await store.save_sources(job, [source], [quote])
        cache = store.selection_cache(job)
        assert await cache.load("batch") is None
        await cache.save("batch", ["fact"])
        assert await store.selection_cache(job).load("batch") == {"ids": ["fact"], "sections": {}}
        episodes, edges = await store.snapshot(user)
        assert await store.publish(
            job, await store.stamp(user), result, retained, episodes, edges, seen_hashes=seen_hashes
        )
        await store.release(job)

        # Invalidating a covered fact revokes it without advancing the worker generation.
        await store.request(user, source_changed=True)
        job = await store.claim(user)
        await store.rows(
            "MATCH ()-[r:RELATES_TO {group_id:$user}]->() SET r.expired_at=datetime()", user=user
        )
        assert await store.read(user) is None
        assert await store.renew(job)
        assert (await store.chat_read(user, result.metadata.compilation_revision))[0] == "revoked"

        # An old worker cannot overwrite a fresh generation's immediate retry.
        await store.request(user, source_changed=True)
        await store.release(job, failed=True)
        job = await store.claim(user)
        assert job and job["attempts"] == 0
        delays = []
        for attempt in range(5):
            before = time.time() * 1000
            await store.release(job, failed=True)
            rows = await store.rows(
                "MATCH (n:SynapseUserCore {user_id:$user}) RETURN "
                "n.hydration_retry_at AS retry,n.hydration_attempts AS attempts",
                user=user,
            )
            delays.append(rows[0]["retry"] - before)
            assert rows[0]["attempts"] == attempt + 1
            await store.rows(
                "MATCH (n:SynapseUserCore {user_id:$user}) SET n.hydration_retry_at=0", user=user
            )
            job = await store.claim(user)
            assert bool(job) == (attempt < 4)
        assert delays[1] > delays[0] + 20000
        await store.request(user)
        assert await store.claim(user) is None
        await store.request(user, source_changed=True)
        assert await store.claim(user)
        print(
            "PASS: real Neo4j publication, additive read/claim race, source writes, selection cache, "
            "destructive revocation, stale release, exponential backoff and terminal recovery"
        )
    finally:
        await store.rows("MATCH (n {group_id:$user}) DETACH DELETE n", user=user)
        await store.rows(DELETE_HYDRATION_QUERY, user=user)
        await driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    asyncio.run(main(parser.parse_args().env_file))
