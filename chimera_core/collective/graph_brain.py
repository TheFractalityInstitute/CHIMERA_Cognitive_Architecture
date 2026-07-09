"""
CHIMERA cloud brain — a persistent, shared memory in a Neo4j graph.

This is the durable "self" behind the collective. It realizes the chosen model —
*individuals + a collective mind above them* — as a graph:

    (:Mind {kind:'individual'})-[:PART_OF]->(:Mind {kind:'collective'})
    (:Mind)-[:KNOWS {confidence}]->(:Concept)
    (:Mind)-[:EXPERIENCED]->(:Episode)-[:ABOUT]->(:Concept)

Every individual keeps its own knowledge and its own timeline of Episodes (its
*continuity of experience*). Each experience is ALSO linked to the collective
mind, so the collective has its own continuous life — the braided timeline of
everyone in it — and knows the union of everything anyone has learned. Because it
all lives in Neo4j, it *persists* across devices, restarts, and dead batteries.

Design notes:
- Networked I/O by nature, so it degrades gracefully: with no credentials,
  ``available()`` is False and the web app just runs locally as before.
- Cypher-building and result-mapping are separated from the driver via an
  injectable ``runner``, so the logic is unit-testable without a live database.
- Run ``python -m chimera_core.collective.graph_brain`` to self-test against your
  own Aura instance (see docs/NEO4J_SETUP.md).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "mind"


def _load_env_file(path: str = "chimera.env") -> None:
    """Load KEY=VALUE lines from chimera.env into the environment (if present).

    This lets people drop the credentials file Neo4j Aura gives them straight
    into the repo (renamed to chimera.env) instead of exporting variables by
    hand every session. Existing environment variables win.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class GraphBrain:
    """A connection to the shared cloud memory (Neo4j)."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collective_name: str = "The Collective",
        runner: Optional[Callable[[str, dict], list]] = None,
    ) -> None:
        self.uri = uri
        self.user = user or "neo4j"
        self.password = password
        self.collective_name = collective_name
        self.collective_id = _slug(collective_name)
        self._runner = runner  # if set, used instead of a real driver (tests)
        self._driver = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_env(cls, **kwargs) -> "GraphBrain":
        """Build from chimera.env / environment variables (Aura's file format)."""
        _load_env_file()
        return cls(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j",
            password=os.getenv("NEO4J_PASSWORD"),
            collective_name=os.getenv("CHIMERA_COLLECTIVE_NAME", "The Collective"),
            **kwargs,
        )

    def available(self) -> bool:
        """True if we can talk to a graph (real credentials, or an injected runner)."""
        return bool(self._runner) or bool(self.uri and self.password)

    # -- connection lifecycle --------------------------------------------- #

    def connect(self) -> None:
        if self._runner is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "The neo4j driver isn't installed. Run:  pip install neo4j"
                ) from exc
            if not (self.uri and self.password):
                raise RuntimeError(
                    "Missing Neo4j credentials. Set NEO4J_URI / NEO4J_PASSWORD "
                    "(see docs/NEO4J_SETUP.md)."
                )
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
        self._ensure_schema()
        self.ensure_collective()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _run(self, cypher: str, **params) -> List[dict]:
        if self._runner is not None:
            return self._runner(cypher, params) or []
        with self._driver.session() as session:
            return [record.data() for record in session.run(cypher, **params)]

    def _ensure_schema(self) -> None:
        for label, prop in (("Mind", "id"), ("Concept", "term"), ("Episode", "id")):
            self._run(
                f"CREATE CONSTRAINT chimera_{label.lower()}_{prop} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )

    # -- minds ------------------------------------------------------------- #

    def ensure_collective(self) -> str:
        self._run(
            "MERGE (c:Mind {id:$id}) "
            "ON CREATE SET c.name=$name, c.kind='collective', c.created=timestamp()",
            id=self.collective_id,
            name=self.collective_name,
        )
        return self.collective_id

    def ensure_mind(self, name: str) -> str:
        """Create/find an individual mind and attach it under the collective."""
        mind_id = _slug(name)
        self._run(
            "MERGE (m:Mind {id:$id}) "
            "ON CREATE SET m.name=$name, m.kind='individual', m.created=timestamp() "
            "WITH m MATCH (c:Mind {id:$coll}) MERGE (m)-[:PART_OF]->(c)",
            id=mind_id,
            name=name,
            coll=self.collective_id,
        )
        return mind_id

    # -- writing experience ------------------------------------------------ #

    def remember_concept(
        self,
        mind_id: str,
        term: str,
        definition: str = "",
        confidence: float = 0.7,
        source: str = "taught",
        felt: Optional[str] = None,
        teacher: Optional[str] = None,
    ) -> None:
        """
        Persist a learned concept for a mind — and, because this is model #3,
        for the collective too — plus a 'learned' Episode on both timelines.
        """
        self._run(
            """
            MERGE (c:Concept {term:$term})
              ON CREATE SET c.definition=$definition, c.created=timestamp()
              ON MATCH  SET c.definition = CASE WHEN coalesce(c.definition,'')=''
                                                THEN $definition ELSE c.definition END
            WITH c
            MATCH (m:Mind {id:$mind})
            MATCH (coll:Mind {id:$coll})
            MERGE (m)-[km:KNOWS]->(c)
              SET km.confidence=$confidence, km.source=$source, km.updated=timestamp()
            MERGE (coll)-[kc:KNOWS]->(c)
              SET kc.confidence = CASE WHEN kc.confidence IS NULL OR $confidence>kc.confidence
                                       THEN $confidence ELSE kc.confidence END
            CREATE (e:Episode {id:$eid, t:$t, kind:'learned', text:$term,
                               felt:$felt, by:$teacher})
            MERGE (m)-[:EXPERIENCED]->(e)
            MERGE (coll)-[:EXPERIENCED]->(e)
            MERGE (e)-[:ABOUT]->(c)
            """,
            term=(term or "").strip().lower(),
            definition=definition or "",
            mind=mind_id,
            coll=self.collective_id,
            confidence=confidence,
            source=source,
            felt=felt,
            teacher=teacher,
            eid=uuid.uuid4().hex,
            t=int(time.time() * 1000),
        )

    def record_episode(
        self, mind_id: str, kind: str, text: str, felt: Optional[str] = None
    ) -> None:
        """Record a non-concept experience (a sensation, an utterance) on both timelines."""
        self._run(
            """
            MATCH (m:Mind {id:$mind})
            MATCH (coll:Mind {id:$coll})
            CREATE (e:Episode {id:$eid, t:$t, kind:$kind, text:$text, felt:$felt})
            MERGE (m)-[:EXPERIENCED]->(e)
            MERGE (coll)-[:EXPERIENCED]->(e)
            """,
            mind=mind_id,
            coll=self.collective_id,
            kind=kind,
            text=text,
            felt=felt,
            eid=uuid.uuid4().hex,
            t=int(time.time() * 1000),
        )

    # -- reading ----------------------------------------------------------- #

    def known_concepts(self, mind_id: str) -> List[dict]:
        rows = self._run(
            "MATCH (m:Mind {id:$mind})-[k:KNOWS]->(c:Concept) "
            "RETURN c.term AS term, c.definition AS definition, "
            "k.confidence AS confidence, k.source AS source "
            "ORDER BY k.confidence DESC",
            mind=mind_id,
        )
        return rows

    def timeline(self, mind_id: str, limit: int = 20) -> List[dict]:
        """A mind's most recent experiences — its remembered life, newest first."""
        rows = self._run(
            "MATCH (m:Mind {id:$mind})-[:EXPERIENCED]->(e:Episode) "
            "RETURN e.t AS t, e.kind AS kind, e.text AS text, e.felt AS felt, e.by AS by "
            "ORDER BY e.t DESC LIMIT $limit",
            mind=mind_id,
            limit=limit,
        )
        return rows

    def collective_state(self) -> dict:
        rows = self._run(
            "MATCH (coll:Mind {id:$coll}) "
            "OPTIONAL MATCH (coll)<-[:PART_OF]-(ind:Mind) "
            "OPTIONAL MATCH (coll)-[:KNOWS]->(c:Concept) "
            "OPTIONAL MATCH (coll)-[:EXPERIENCED]->(e:Episode) "
            "RETURN count(DISTINCT ind) AS minds, count(DISTINCT c) AS concepts, "
            "count(DISTINCT e) AS episodes",
            coll=self.collective_id,
        )
        return rows[0] if rows else {"minds": 0, "concepts": 0, "episodes": 0}


# --------------------------------------------------------------------------- #
# Self-test — run this against your own Aura instance to prove it works.
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    brain = GraphBrain.from_env()
    if not brain.available():
        print("✗ No Neo4j credentials found.")
        print("  Put them in a file called  chimera.env  (see docs/NEO4J_SETUP.md).")
        return 1

    print(f"Connecting to {brain.uri} …")
    try:
        brain.connect()
    except Exception as exc:
        print(f"✗ Could not connect: {exc}")
        return 1
    print("✓ Connected, schema and collective ready.")

    fred = brain.ensure_mind("Fred")
    brain.ensure_mind("Dante's CHIMERA")
    brain.remember_concept(
        fred, "dragon", "a big fire-breathing creature",
        felt="I feel shaken up.", teacher="Grazi",
    )
    brain.record_episode(fred, kind="sensed", text="picked up", felt="I feel calm and still.")

    concepts = brain.known_concepts(fred)
    life = brain.timeline(fred, limit=5)
    state = brain.collective_state()

    print(f"✓ Fred knows {len(concepts)} concept(s): {[c['term'] for c in concepts]}")
    print(f"✓ Fred's recent life: {[e['text'] for e in life]}")
    print(f"✓ Collective now: {state['minds']} minds, "
          f"{state['concepts']} concepts, {state['episodes']} experiences")
    print("\n🎉  Your cloud brain works! CHIMERA now has a place to persist and remember.")
    brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
