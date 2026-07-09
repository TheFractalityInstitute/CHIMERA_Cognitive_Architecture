"""
CHIMERA Collective — in-process coordinator for the web UI.

``LocalCollective`` hosts several CHIMERA nodes inside a single process (one per
person/browser) around a shared ``CollectiveHub``. It is what lets a family run
*one* app and have each member's CHIMERA pool its learning with the others: teach
a word on one node and this class propagates it to every other node, returning a
list of "who just learned what from whom" events the web layer can announce.

It is deliberately transport-free (no Flask, no sockets) so it can be unit
tested. ``web/app.py`` is a thin wrapper over it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from chimera_core.collective.client import CollectiveNode
from chimera_core.collective.hub import CollectiveHub
from chimera_core.language.chimera_language_learning import OrganicLearningSystem


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "node"


class LocalCollective:
    """Manages many in-process CHIMERA nodes sharing one collective."""

    def __init__(self, persist_dir: Optional[str | Path] = None) -> None:
        self.hub = CollectiveHub()
        self.nodes: Dict[str, CollectiveNode] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    # -- node lifecycle ---------------------------------------------------- #

    def get_or_create(self, name: str) -> CollectiveNode:
        if name not in self.nodes:
            learning = OrganicLearningSystem(name)
            self._load(name, learning)
            self.nodes[name] = CollectiveNode(learning, name)
            self.hub.register_node(name, node_id=name)
        return self.nodes[name]

    def has_node(self, name: str) -> bool:
        return name in self.nodes

    def roster(self) -> List[str]:
        return list(self.nodes.keys())

    # -- learning ---------------------------------------------------------- #

    def teach(self, name: str, concept: str, explanation: str, examples=None) -> dict:
        """Teach a concept to one node (does not share it — call ``share`` for that)."""
        node = self.get_or_create(name)
        result = node.learning.teach(concept, explanation, examples)
        self._save(name)
        return result

    def share(self, name: str, term: str) -> List[dict]:
        """
        Share a term from node ``name`` to the collective and let every other node
        absorb it. Returns one event dict per node that *newly* learned it:
        ``{"to": <node>, "from": <node>, "term": ..., "definition": ...}``.
        """
        node = self.get_or_create(name)
        payload = node.make_payload(term)
        if not payload:
            return []

        concept = self.hub.share_concept(name, term, payload["definition"], payload["examples"])
        if concept is None:
            return []

        concept_dict = concept.to_dict()
        concept_dict["contributor"] = name

        events: List[dict] = []
        for other_name, other in self.nodes.items():
            if other_name == name:
                continue
            if other.absorb(concept_dict):
                self._save(other_name)
                events.append(
                    {
                        "to": other_name,
                        "from": name,
                        "term": concept.term,
                        "definition": concept_dict["definition"],
                    }
                )
        return events

    def teach_and_share(self, name: str, concept: str, explanation: str, examples=None):
        """Convenience: teach a concept and immediately share it. Returns (result, events)."""
        result = self.teach(name, concept, explanation, examples)
        events = self.share(name, concept)
        return result, events

    # -- views ------------------------------------------------------------- #

    def state(self) -> dict:
        return self.hub.state()

    def concepts_for(self, name: str) -> List[dict]:
        node = self.nodes.get(name)
        if not node:
            return []
        out = []
        for term, entry in node.learning.language.vocabulary.items():
            out.append(
                {
                    "term": term,
                    "confidence": entry.get("confidence", 0),
                    "source": entry.get("source", "self"),
                    "contributor": entry.get("contributor"),
                    "taught": entry.get("taught", False),
                }
            )
        out.sort(key=lambda x: x["confidence"], reverse=True)
        return out

    # -- persistence ------------------------------------------------------- #

    def _path(self, name: str) -> Optional[Path]:
        return self.persist_dir / f"{_slug(name)}.json" if self.persist_dir else None

    def _load(self, name: str, learning: OrganicLearningSystem) -> None:
        path = self._path(name)
        if path and path.exists():
            try:
                learning.load_state(str(path))
            except Exception:
                pass  # a corrupt/old snapshot shouldn't block a fresh start

    def _save(self, name: str) -> None:
        path = self._path(name)
        node = self.nodes.get(name)
        if path and node:
            try:
                node.learning.save_state(str(path))
            except Exception:
                pass
