"""
CHIMERA Collective — the node side.

A ``CollectiveNode`` wraps a single device's OrganicLearningSystem and knows how
to (a) turn one of its learned concepts into a shareable payload and (b) absorb a
concept shared by the collective into its own vocabulary. This logic is
transport-free so it can be unit-tested and reused.

``SocketNodeClient`` connects a CollectiveNode to a real hub over Socket.IO.
"""

from __future__ import annotations

from typing import List, Optional

from chimera_core.language.chimera_language_learning import OrganicLearningSystem


class CollectiveNode:
    """Bridges one OrganicLearningSystem to the collective's shared-knowledge model."""

    def __init__(self, learning: OrganicLearningSystem, name: str) -> None:
        self.learning = learning
        self.name = name
        # Terms this node has taken from the collective — so we can attribute
        # them and avoid re-absorbing our own echoes.
        self.absorbed: set[str] = set()

    # -- outbound ---------------------------------------------------------- #

    def make_payload(self, term: str) -> Optional[dict]:
        """Build a shareable payload for a term this node knows, or None."""
        term = (term or "").strip().lower()
        entry = self.learning.language.vocabulary.get(term)
        if not entry:
            return None
        meanings = entry.get("meanings") or []
        return {
            "term": term,
            "definition": meanings[0] if meanings else "",
            "examples": entry.get("examples", []),
            "contributor": self.name,
        }

    def shareable_concepts(self) -> List[dict]:
        """Every taught concept with a definition — the node's contribution."""
        payloads = []
        for term, entry in self.learning.language.vocabulary.items():
            if entry.get("taught") and (entry.get("meanings") or []):
                payload = self.make_payload(term)
                if payload:
                    payloads.append(payload)
        return payloads

    # -- inbound ----------------------------------------------------------- #

    def absorb(self, concept: dict) -> bool:
        """
        Learn a concept handed down by the collective. Returns True if it was
        newly absorbed. Skips concepts this node itself contributed.
        """
        term = (concept.get("term") or "").strip().lower()
        if not term:
            return False
        if concept.get("contributor") == self.name:
            return False  # don't re-absorb our own contribution echoed back

        existing = self.learning.language.vocabulary.get(term)
        newly = term not in self.absorbed and not (existing and existing.get("taught"))

        self.learning.teach(term, concept.get("definition", ""), concept.get("examples") or [])
        entry = self.learning.language.vocabulary.get(term)
        if entry is not None:
            entry["source"] = "collective"
            entry["contributor"] = concept.get("contributor")

        self.absorbed.add(term)
        return newly


class SocketNodeClient:
    """Connects a CollectiveNode to a hub via Socket.IO."""

    def __init__(self, node: CollectiveNode, hub_url: str = "http://localhost:5001") -> None:
        import socketio  # python-socketio client

        self.node = node
        self.hub_url = hub_url
        self.node_id: Optional[str] = None
        self.sio = socketio.Client()

        @self.sio.on("welcome")
        def _welcome(data):
            self.node_id = data.get("node_id")

        @self.sio.on("collective_snapshot")
        def _snapshot(data):
            for concept in data.get("concepts", []):
                self.node.absorb(concept)

        @self.sio.on("collective_concept")
        def _concept(data):
            self.node.absorb(data.get("concept", {}))

    def connect(self) -> None:
        self.sio.connect(self.hub_url, auth={"name": self.node.name})

    def disconnect(self) -> None:
        self.sio.disconnect()

    def share(self, term: str) -> bool:
        """Share one learned term with the collective."""
        payload = self.node.make_payload(term)
        if not payload:
            return False
        self.sio.emit("share_concept", payload)
        return True

    def share_all(self) -> int:
        """Share every taught concept this node knows. Returns the count shared."""
        payloads = self.node.shareable_concepts()
        for payload in payloads:
            self.sio.emit("share_concept", payload)
        return len(payloads)
