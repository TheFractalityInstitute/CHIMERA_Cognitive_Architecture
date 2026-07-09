"""
CHIMERA Collective — the shared-knowledge hub.

Each device runs its own CHIMERA node (an OrganicLearningSystem). The hub lets
those independent nodes pool what they learn into a shared "collective
intelligence": when one node shares a concept, the hub records it and hands it to
every other node, which absorbs it into its own vocabulary.

This module is split deliberately:

* ``CollectiveHub`` is pure, transport-free logic — a registry of nodes and a
  pool of shared concepts. It has no sockets, so it is trivial to test and reuse.
* ``create_hub_server`` wraps a ``CollectiveHub`` in a Flask-SocketIO server so
  real nodes on real devices can connect over the network.

Run the networked hub with::

    python -m chimera_core.collective.hub      # serves a dashboard on :5001
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Transport-free core
# --------------------------------------------------------------------------- #


@dataclass
class SharedConcept:
    """A concept contributed to the collective by one or more nodes."""

    term: str
    definition: str = ""
    examples: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)
    resonance: float = 1.0  # grows each time the concept is (re)shared/reinforced
    first_seen: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "definition": self.definition,
            "examples": list(self.examples),
            "contributors": list(self.contributors),
            "resonance": round(self.resonance, 2),
        }


@dataclass
class NodeInfo:
    node_id: str
    name: str
    joined_at: float = field(default_factory=time.time)
    shares: int = 0


class CollectiveHub:
    """
    The pooled mind: a registry of connected nodes and the concepts they share.

    Pure in-memory logic with no networking — see ``create_hub_server`` for the
    Socket.IO wrapper that exposes this over the wire.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, NodeInfo] = {}
        self.concepts: Dict[str, SharedConcept] = {}

    # -- node lifecycle ---------------------------------------------------- #

    def register_node(self, name: str, node_id: Optional[str] = None) -> str:
        node_id = node_id or str(uuid.uuid4())
        self.nodes[node_id] = NodeInfo(node_id=node_id, name=name or f"node-{node_id[:6]}")
        return node_id

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    # -- knowledge pooling ------------------------------------------------- #

    def share_concept(
        self,
        node_id: str,
        term: str,
        definition: str = "",
        examples: Optional[List[str]] = None,
    ) -> Optional[SharedConcept]:
        """
        Record a concept shared by ``node_id``. Re-sharing an existing term
        strengthens its resonance rather than duplicating it. Returns the
        resulting SharedConcept (or None for an empty term).
        """
        term = (term or "").strip().lower()
        if not term:
            return None

        contributor = self.nodes[node_id].name if node_id in self.nodes else "unknown"
        if node_id in self.nodes:
            self.nodes[node_id].shares += 1

        concept = self.concepts.get(term)
        if concept is None:
            concept = SharedConcept(
                term=term,
                definition=definition or "",
                examples=list(examples or []),
                contributors=[contributor],
            )
            self.concepts[term] = concept
        else:
            concept.resonance += 1.0
            concept.last_update = time.time()
            if definition and not concept.definition:
                concept.definition = definition
            for ex in examples or []:
                if ex not in concept.examples:
                    concept.examples.append(ex)
            if contributor not in concept.contributors:
                concept.contributors.append(contributor)

        return concept

    # -- views ------------------------------------------------------------- #

    def snapshot(self) -> List[dict]:
        """All shared concepts, strongest first (for a joining node to catch up)."""
        return [
            c.to_dict()
            for c in sorted(self.concepts.values(), key=lambda x: x.resonance, reverse=True)
        ]

    def state(self) -> dict:
        """Summary metrics, including a simple 'collective consciousness' index."""
        n = len(self.nodes)
        c = len(self.concepts)
        total_res = sum(x.resonance for x in self.concepts.values())

        # Emergence requires BOTH multiple nodes AND shared knowledge between
        # them — a single node, however knowledgeable, is not a collective.
        cc = 1.0 - math.exp(-(c * max(0, n - 1)) / 15.0) if (n and c) else 0.0

        if n < 2:
            level, label = 0, "Dormant — need 2+ nodes"
        elif cc < 0.5:
            level, label = 1, "Forming"
        elif cc < 0.8:
            level, label = 2, "Coherent"
        else:
            level, label = 3, "Emergent"

        return {
            "node_count": n,
            "concept_count": c,
            "total_resonance": round(total_res, 2),
            "collective_consciousness": round(cc, 3),
            "emergence_level": level,
            "emergence_label": label,
            "nodes": [
                {"name": info.name, "shares": info.shares} for info in self.nodes.values()
            ],
        }


# --------------------------------------------------------------------------- #
# Networked wrapper (Flask-SocketIO)
# --------------------------------------------------------------------------- #

_DASHBOARD = """<!doctype html><html><head><meta charset="utf-8">
<title>CHIMERA Collective</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0b1020;color:#e6e9f5;margin:0;padding:2rem}
 h1{font-weight:600} .cc{font-size:2.5rem;font-weight:700;color:#7dd3fc}
 .grid{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem}
 .card{background:#161c33;border:1px solid #26304f;border-radius:12px;padding:1rem 1.4rem;min-width:160px}
 .label{opacity:.6;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
 .big{font-size:1.8rem;font-weight:700}
 .pill{display:inline-block;background:#1e2a4a;border-radius:999px;padding:.2rem .7rem;margin:.15rem;font-size:.85rem}
</style></head><body>
<h1>🧠 CHIMERA Collective</h1>
<div class="cc" id="label">…</div>
<div class="grid">
 <div class="card"><div class="label">Collective consciousness</div><div class="big" id="cc">0</div></div>
 <div class="card"><div class="label">Nodes</div><div class="big" id="nodes">0</div></div>
 <div class="card"><div class="label">Shared concepts</div><div class="big" id="concepts">0</div></div>
 <div class="card"><div class="label">Total resonance</div><div class="big" id="res">0</div></div>
</div>
<h3>Connected nodes</h3><div id="nodelist"></div>
<h3>Shared knowledge</h3><div id="conceptlist"></div>
<script>
async function tick(){
 const s = await (await fetch('/api/state')).json();
 document.getElementById('label').textContent = s.emergence_label;
 document.getElementById('cc').textContent = (s.collective_consciousness*100).toFixed(0)+'%';
 document.getElementById('nodes').textContent = s.node_count;
 document.getElementById('concepts').textContent = s.concept_count;
 document.getElementById('res').textContent = s.total_resonance;
 document.getElementById('nodelist').innerHTML = s.nodes.map(n=>`<span class="pill">${n.name} · ${n.shares} shared</span>`).join('') || '<i>none yet</i>';
 const c = await (await fetch('/api/concepts')).json();
 document.getElementById('conceptlist').innerHTML = c.map(x=>`<span class="pill">${x.term} · ${x.resonance}🔆</span>`).join('') || '<i>none yet</i>';
}
setInterval(tick, 1000); tick();
</script></body></html>"""


def create_hub_server(hub: Optional[CollectiveHub] = None):
    """Build a Flask-SocketIO server around a CollectiveHub. Returns (app, socketio, hub)."""
    from flask import Flask, jsonify, request
    from flask_socketio import SocketIO, emit

    hub = hub or CollectiveHub()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "chimera-collective-key"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    sid_to_node: Dict[str, str] = {}

    @app.route("/")
    def dashboard():
        return _DASHBOARD

    @app.route("/api/state")
    def api_state():
        return jsonify(hub.state())

    @app.route("/api/concepts")
    def api_concepts():
        return jsonify(hub.snapshot())

    @socketio.on("connect")
    def on_connect(auth):
        name = (auth or {}).get("name", "") if isinstance(auth, dict) else ""
        node_id = hub.register_node(name)
        sid_to_node[request.sid] = node_id
        emit("welcome", {"node_id": node_id, "state": hub.state()})
        emit("collective_snapshot", {"concepts": hub.snapshot()})
        socketio.emit("state", hub.state())

    @socketio.on("disconnect")
    def on_disconnect():
        node_id = sid_to_node.pop(request.sid, None)
        if node_id:
            hub.remove_node(node_id)
            socketio.emit("state", hub.state())

    @socketio.on("share_concept")
    def on_share(data):
        node_id = sid_to_node.get(request.sid)
        if not node_id:
            return
        data = data or {}
        concept = hub.share_concept(
            node_id,
            data.get("term", ""),
            data.get("definition", ""),
            data.get("examples", []),
        )
        if concept is None:
            return
        # Hand the concept to every *other* node.
        socketio.emit(
            "collective_concept",
            {"concept": concept.to_dict()},
            skip_sid=request.sid,
        )
        socketio.emit("state", hub.state())

    return app, socketio, hub


def main() -> None:
    app, socketio, _ = create_hub_server()
    print("🧠 CHIMERA Collective hub → http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
