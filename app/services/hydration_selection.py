"""Bounded graph selection. Rank metadata first; read fact text only after each LIMIT."""

from app.services.hydration_result import GraphStats

CORRECTION_EPISODES = 20
CORRECTION_FACTS = 60
RECENT_EPISODES = 30
RECENT_FACTS = 150
HUBS = 15
FACTS_PER_HUB = 8
DIVERSE_HUBS = 2
FACTS_PER_DIVERSE_HUB = 400
MAX_FACTS = 800
INPUT_CHARS = 120_000
CORRECTION_NAMES = ["user_memory_correction", "notion_correction"]

# localdatetime normalizes both Graphiti's naive timestamps and legacy zoned values.
ACTIVE = """r.expired_at IS NULL AND
    (r.invalid_at IS NULL OR localdatetime({datetime:r.invalid_at})>localdatetime())"""
FIELDS = """r.uuid AS uuid,r.fact AS fact,
    toString(r.valid_at) AS valid_at,toString(r.invalid_at) AS invalid_at,
    a.uuid AS source_uuid,b.uuid AS target_uuid"""
CORRECTION = """EXISTS {MATCH (c:Episodic {group_id:$user})
    WHERE c.source='text' AND c.name IN $names AND c.uuid IN r.episodes}"""
HEAD_QUERY = """
MATCH (e:Episodic {group_id:$user})
WITH e ORDER BY e.created_at DESC,e.uuid DESC LIMIT 1
RETURN e.uuid AS uuid
"""
RECENT_QUERY = (
    """
MATCH (e:Episodic {group_id:$user})
WHERE NOT $corrections OR (e.source='text' AND e.name IN $names)
WITH e ORDER BY localdatetime({datetime:e.valid_at}) DESC,e.uuid LIMIT $episode_limit
UNWIND coalesce(e.entity_edges,[]) AS id
MATCH (a)-[r:RELATES_TO {uuid:id}]->(b)
WHERE r.group_id=$user AND a.group_id=$user AND b.group_id=$user AND """
    + ACTIVE
    + """
WITH a,b,r,max(toString(e.valid_at)) AS date,collect(DISTINCT e.uuid) AS episode_ids
WITH a,b,r,date,episode_ids,"""
    + CORRECTION
    + """ AS correction
ORDER BY correction DESC,date DESC,r.uuid LIMIT $fact_limit
RETURN """
    + FIELDS
    + """,date,episode_ids,correction
"""
)
LEGACY_QUERY = (
    """
MATCH (a)-[r:RELATES_TO {group_id:$user}]->(b)
WHERE a.group_id=$user AND b.group_id=$user AND NOT r.uuid IN $excluded AND """
    + ACTIVE
    + """
WITH a,b,r ORDER BY r.created_at DESC,r.uuid LIMIT $limit
RETURN """
    + FIELDS
    + """,toString(r.created_at) AS date,[] AS episode_ids,
"""
    + CORRECTION
    + """ AS correction
"""
)
HUB_QUERY = (
    """
MATCH (n:Entity {group_id:$user})
WITH n,COUNT{(n)-[:RELATES_TO]-()} AS degree
ORDER BY degree DESC,n.uuid LIMIT $hub_candidates
CALL {
    WITH n
    MATCH (a)-[r:RELATES_TO]-(n)
    WHERE r.group_id=$user AND a.group_id=$user AND NOT r.uuid IN $excluded AND """
    + ACTIVE
    + """
    WITH n,a,r,COUNT{(a)-[:RELATES_TO]-()} AS neighbor_degree
    ORDER BY neighbor_degree DESC,r.created_at DESC,r.uuid LIMIT $per_hub
    WITH n,a,r
    RETURN collect({uuid:r.uuid,fact:r.fact,valid_at:toString(r.valid_at),
        invalid_at:toString(r.invalid_at),date:toString(r.created_at),episode_ids:[],
        source_uuid:startNode(r).uuid,target_uuid:endNode(r).uuid,
        correction:"""
    + CORRECTION
    + """}) AS facts
}
WITH n,degree,facts WHERE size(facts)>0
ORDER BY degree DESC,n.uuid LIMIT $hubs
RETURN facts
"""
)
# Round-robin across (neighbor, relation kind) before repeating a pair.
# The ordering reads relationship metadata; fact text is projected after LIMIT.
DIVERSITY_QUERY = (
    """
MATCH (n:Entity {group_id:$user})
WITH n,COUNT{(n)-[:RELATES_TO]-()} AS degree
ORDER BY degree DESC,n.uuid LIMIT $hub_candidates
CALL {
    WITH n
    MATCH (n)-[r:RELATES_TO]-(a)
    WHERE r.group_id=$user AND a.group_id=$user AND NOT r.uuid IN $excluded AND """
    + ACTIVE
    + """
    WITH r,a.uuid AS neighbor,coalesce(r.name,'') AS kind
    ORDER BY r.created_at DESC,r.uuid
    WITH neighbor,kind,collect(r) AS members
    UNWIND range(0,size(members)-1) AS position
    WITH members[position] AS r,position
    ORDER BY position,r.created_at DESC,r.uuid LIMIT $per_hub
    RETURN collect({uuid:r.uuid,fact:r.fact,valid_at:toString(r.valid_at),
        invalid_at:toString(r.invalid_at),date:toString(r.created_at),episode_ids:[],
        source_uuid:startNode(r).uuid,target_uuid:endNode(r).uuid,
        correction:"""
    + CORRECTION
    + """}) AS facts
}
WITH n,degree,facts WHERE size(facts)>0
ORDER BY degree DESC,n.uuid LIMIT $hubs
RETURN facts
"""
)
VALIDATE_QUERY = (
    """
UNWIND $ids AS id
MATCH (a)-[r:RELATES_TO {uuid:id}]->(b)
WHERE r.group_id=$user AND a.group_id=$user AND b.group_id=$user AND """
    + ACTIVE
    + """
RETURN """
    + FIELDS
)
STATS_QUERY = (
    """CALL {MATCH (n:Entity {group_id:$user}) RETURN count(n) AS entities}
    CALL {MATCH ()-[r:RELATES_TO {group_id:$user}]->() WHERE """
    + ACTIVE
    + """
          RETURN count(r) AS relationships}
    RETURN entities,relationships"""
)


async def select(store, user):
    head = await store.head(user)
    selected = {}
    for corrections, episodes, facts in (
        (True, CORRECTION_EPISODES, CORRECTION_FACTS),
        (False, RECENT_EPISODES, RECENT_FACTS),
    ):
        rows = await store.rows(
            RECENT_QUERY,
            user=user,
            corrections=corrections,
            names=CORRECTION_NAMES,
            episode_limit=episodes,
            fact_limit=facts,
        )
        for row in rows:
            cid = row["uuid"]
            if cid in selected:
                selected[cid]["episode_ids"] = sorted(
                    set(selected[cid]["episode_ids"] + row["episode_ids"])
                )
            else:
                selected[cid] = {
                    **row,
                    "priority": 0 if corrections else 1,
                    "date_kind": "episodio",
                }
    recent_count = sum(c["priority"] == 1 for c in selected.values())
    if recent_count < RECENT_FACTS:
        for row in await store.rows(
            LEGACY_QUERY,
            user=user,
            names=CORRECTION_NAMES,
            excluded=list(selected),
            limit=RECENT_FACTS - recent_count,
        ):
            selected[row["uuid"]] = {**row, "priority": 1, "date_kind": "registro"}
    for query, hubs, per_hub, priority in (
        (HUB_QUERY, HUBS, FACTS_PER_HUB, 2),
        (DIVERSITY_QUERY, DIVERSE_HUBS, FACTS_PER_DIVERSE_HUB, 3),
    ):
        rows = await store.rows(
            query,
            user=user,
            names=CORRECTION_NAMES,
            excluded=list(selected),
            hub_candidates=hubs * 2,
            hubs=hubs,
            per_hub=per_hub,
        )
        for hub in rows:
            for row in hub["facts"]:
                selected.setdefault(
                    row["uuid"], {**row, "priority": priority, "date_kind": "registro"}
                )
    stats = (await store.rows(STATS_QUERY, user=user))[0]
    return (
        list(selected.values()),
        head,
        GraphStats(stats["entities"], stats["relationships"], total_chars=None),
    )
