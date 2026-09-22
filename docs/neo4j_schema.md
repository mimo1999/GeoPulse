# Neo4j Graph Schema — Actor-Interaction Graph

Design for usecase.md's core deliverable: *"Transform GDELT event data into a graph model
representing actors and their interactions over time... analyze the resulting interaction
network to identify patterns... and how they evolve over time."*

This is a **design**, not yet a migration — nothing has been loaded into Neo4j. It defines the
target schema and explains the choices, so the migration (separate, later work) has something
deliberate to build against instead of a mechanical 1:1 port of the Postgres `graph.*` tables.

---

## Source of truth today

`graph.*` in Postgres (built this session, `ingestion/graph_builder.py`): 48.9M events, 79.6M
event-actor edges, 51,849 actors, 245,075 locations, over 2023-2025 GDELT data. That relational
schema already did the hard entity-resolution work (actor identity, country inference, CAMEO
interaction typing) — Neo4j's job is to host it as a graph for the traversal/network queries a
relational table can't do cheaply, not to redo that resolution.

**Design principle: carry Postgres's already-correct synthetic IDs as Neo4j node identity
properties, with uniqueness constraints.** `graph.actor`'s natural key is `(source, raw_code,
country_iso3)`, but `country_iso3` is nullable — a NODE KEY constraint (which requires every
listed property to be present) can't express that cleanly. Postgres has already resolved this
correctly; re-deriving it in Cypher would just risk reintroducing the exact bugs (actor
collapse, silent NULL handling) that `graph_builder.py`'s design already avoided. So: import
`actor_id`, `event_id`, `location_id` verbatim as the node keys.

---

## Node labels

| Label | Key property | Other properties | Source |
|---|---|---|---|
| `Country` | `iso3` (unique) | `name` | `graph.country` |
| `Actor` | `actor_id` (unique) | `source`, `raw_code`, `raw_name`, `country_inferred` | `graph.actor` |
| `Location` | `location_id` (unique) | `lat`, `lon` | `graph.location` |
| `Event` | `event_id` (unique) | `source`, `source_event_id`, `event_date` (native Neo4j `date`), `event_type` (raw CAMEO root code), `interaction_type`, `intensity` (Goldstein), `avg_tone`, `num_mentions`, `fatalities_best/low/high`, `source_url` | `graph.event` |

`event_date` is stored as a native Neo4j temporal type (`date($event_date)` at load time), not a
string — this is what makes "how they evolve over time" queries (range filters, time-bucketed
aggregation) native Cypher instead of string parsing.

## Relationships

| Relationship | Direction | Properties | Meaning |
|---|---|---|---|
| `(:Actor)-[:INITIATED]->(:Event)` | Actor → Event | — | GDELT's `actor1` (the source of the action) |
| `(:Event)-[:TARGETED]->(:Actor)` | Event → Actor | — | GDELT's `actor2` (the recipient), **optional** — 35.3M of 79.6M event-actor rows are `actor2`; the rest (44.3M `actor1`-only) are unilateral statements with no target. An `Event` with no outgoing `TARGETED` is a valid, expected shape, not a data gap. |
| `(:Event)-[:OCCURRED_AT]->(:Location)` | Event → Location | — | |
| `(:Location)-[:IN_COUNTRY]->(:Country)` | Location → Country | — | |
| `(:Actor)-[:BELONGS_TO]->(:Country)` | Actor → Country | — | only when `country_iso3` is resolved (native or CAMEO-role-inferred); a genuinely unresolved actor has no outgoing `BELONGS_TO` — this is usecase.md's own named limitation ("generic references... too general"), and it should be an absent edge a Cypher query can detect, not a guessed one. |
| `(:Actor)-[:CANONICAL_OF]->(:Actor)` | alias → canonical | — | entity-resolution linking (`canonical_actor_id`). Currently unused (0 rows in Postgres) — schema-supported, dormant. |

### Why Event stays a node (not collapsed into a direct Actor→Actor edge)

The alternative — a single `(:Actor)-[:INTERACTED_WITH {date, interaction_type, ...}]->(:Actor)`
relationship per event — is the more common pattern for pure interaction-network datasets, and
would make Actor-to-Actor traversals one hop instead of two. It was rejected for this schema
**because it would throw away exactly the fields a CI analyst needs to trust an edge**:
`source_url` (provenance — "why does this connection exist?"), the geolocation
(`OCCURRED_AT`/`IN_COUNTRY`), and the option for an event with zero or one actor (a unilateral
statement isn't representable as an edge between two nodes at all). Losing those to save one
traversal hop is the wrong trade for a Competitive Intelligence tool whose stated limitation is
already "context can be too generic" — this schema should make context easier to recover, not
harder.

### How network analysis still gets a fast Actor-Actor view

Rather than *also* persisting a redundant, ~80M-row `INTERACTED_WITH` edge type (doubling
storage for a view that's mechanically derivable), use **Neo4j Graph Data Science (GDS)**
projections: `gds.graph.project.cypher(...)` builds an in-memory monopartite Actor-Actor graph
on demand from the `INITIATED`/`TARGETED` paths (optionally filtered by `interaction_type` or a
date range), and algorithms (PageRank, Louvain community detection, weakly-connected
components) run against that projection, writing results back as `Actor` properties
(`pagerank_score`, `community_id`). This is the idiomatic Neo4j answer to "I have a bipartite
event graph but want to analyze a derived actor network," and it avoids a second copy of the
data going stale relative to the first. GDS is a separate plugin (like APOC) — not yet installed
on the local Desktop instance; needed before the analysis step, not before the schema/migration.

If ad hoc Cypher without GDS turns out to be the dominant access pattern (simpler queries, no
plugin dependency), a materialized `INTERACTED_WITH` edge is a reasonable fallback — noted here
so the choice is revisited with real query patterns in hand, not decided blind.

---

## Constraints and indexes

The local Desktop instance reports as Enterprise edition (Desktop's dev license), so full
`NODE KEY` constraints are available, not just uniqueness:

```cypher
CREATE CONSTRAINT country_key IF NOT EXISTS FOR (c:Country) REQUIRE c.iso3 IS NODE KEY;
CREATE CONSTRAINT actor_key   IF NOT EXISTS FOR (a:Actor)   REQUIRE a.actor_id IS NODE KEY;
CREATE CONSTRAINT location_key IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS NODE KEY;
CREATE CONSTRAINT event_key   IF NOT EXISTS FOR (e:Event)   REQUIRE e.event_id IS NODE KEY;

CREATE INDEX event_date_idx        IF NOT EXISTS FOR (e:Event) ON (e.event_date);
CREATE INDEX event_interaction_idx IF NOT EXISTS FOR (e:Event) ON (e.interaction_type);
CREATE INDEX actor_country_idx     IF NOT EXISTS FOR (a:Actor) ON (a.country_iso3);
```

(`country_iso3` stays a plain indexed property on `Actor`, not part of a constraint — it's
nullable, and the `BELONGS_TO` relationship is the real source of truth for actor-country
membership; the property is a convenience for filtering without a traversal.)

---

## Example queries this schema is built to answer

**Which countries had the most conflict-typed interactions with China in 2024?**
```cypher
MATCH (a1:Actor)-[:BELONGS_TO]->(:Country {iso3: 'CHN'})
MATCH (a1)-[:INITIATED]->(e:Event {interaction_type: 'conflict'})-[:TARGETED]->(a2:Actor)
MATCH (a2)-[:BELONGS_TO]->(c2:Country)
WHERE e.event_date >= date('2024-01-01') AND e.event_date < date('2025-01-01')
RETURN c2.iso3, count(*) AS conflict_events
ORDER BY conflict_events DESC LIMIT 10;
```

**Monthly interaction-type mix between two countries, over time (the "how it evolves" ask):**
```cypher
MATCH (a1:Actor)-[:BELONGS_TO]->(:Country {iso3: 'USA'})
MATCH (a1)-[:INITIATED]->(e:Event)-[:TARGETED]->(a2:Actor)-[:BELONGS_TO]->(:Country {iso3: 'CHN'})
RETURN e.event_date.year AS year, e.event_date.month AS month,
       e.interaction_type, count(*) AS n
ORDER BY year, month;
```

**Trace an edge back to its source article (the brief's own suggested fix for generic actors):**
```cypher
MATCH (a1:Actor)-[:INITIATED]->(e:Event)-[:TARGETED]->(a2:Actor)
WHERE a1.actor_id = $id1 AND a2.actor_id = $id2
RETURN e.event_date, e.interaction_type, e.source_url
ORDER BY e.event_date DESC LIMIT 20;
```

---

## Known gaps to resolve before migrating data (not before finalizing the schema)

1. **`interaction_type` is NULL on the 1.1M events from the untracked June-2026 prototype**
   (predates the column). Cheap to backfill in Postgres first via `data/cameo_codes.py` against
   the stored `event_type`/root code — recommended before migration rather than importing NULLs.
2. **GDS plugin** not yet installed on the Desktop instance — needed for the network-analysis
   step, not for loading the base graph.
3. **Migration path** (Postgres → Neo4j) is separate work: likely `apoc.periodic.iterate` batches
   or an external ETL script (`neo4j_client.py`, already built, is the connection layer for
   whichever approach). Not scoped here — this document is the target shape, not the loader.
