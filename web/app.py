"""
CHIMERA web application — family collective edition.

One app, many minds. Each browser names its own CHIMERA node; all nodes live in
this one process and share a collective, so teaching a word on one node flows to
everyone else's node live. No terminal needed to use it — just open the page.

Start it with:  python run.py   → http://localhost:5000
Others on the same WiFi can join at  http://<your-computer-ip>:5000
"""

import asyncio
import atexit
import threading
from collections import defaultdict
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimera_core.collective.local import LocalCollective

# --------------------------------------------------------------------------- #
# App + shared state
# --------------------------------------------------------------------------- #

app = Flask(__name__)
app.config["SECRET_KEY"] = "chimera-secret-key-change-in-production"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nodes"
collective = LocalCollective(persist_dir=DATA_DIR)

sessions: dict[str, str] = {}                      # sid -> node name
name_sids: dict[str, set] = defaultdict(set)       # node name -> set of sids

# A dedicated asyncio loop for the (async) interact() calls.
_loop = asyncio.new_event_loop()


def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop, daemon=True).start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


atexit.register(lambda: [collective._save(n) for n in collective.roster()])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def node_stats(name: str) -> dict:
    node = collective.get_or_create(name)
    ls = node.learning
    return {
        "name": name,
        "thoughts": len(ls.reasoning.thought_network),
        "concepts": len(ls.language.vocabulary),
        "conversations": ls.conversation_count,
        "development_stage": ls._get_development_stage(),
    }


def broadcast_collective():
    socketio.emit("collective_state", collective.state())
    socketio.emit("roster", {"nodes": collective.roster()})


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/concepts")
def api_concepts():
    name = request.args.get("name", "")
    return jsonify(collective.concepts_for(name))


@app.route("/api/collective")
def api_collective():
    return jsonify(collective.state())


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


@socketio.on("join")
def on_join(data):
    name = ((data or {}).get("name") or "").strip() or "A CHIMERA"
    sessions[request.sid] = name
    name_sids[name].add(request.sid)
    join_room(name)

    collective.get_or_create(name)
    emit("joined", {"name": name, "stats": node_stats(name)})
    broadcast_collective()


@socketio.on("disconnect")
def on_disconnect():
    name = sessions.pop(request.sid, None)
    if name:
        name_sids[name].discard(request.sid)
        leave_room(name)


@socketio.on("message")
def on_message(data):
    name = sessions.get(request.sid)
    if not name:
        return
    text = ((data or {}).get("text") or "").strip()
    if not text:
        return

    node = collective.get_or_create(name)
    result = run_async(node.learning.interact(text))

    emit(
        "response",
        {
            "text": result["response"],
            "confidence": result["understanding"],
            "thoughts_formed": result["thoughts_formed"],
            "words_known": result["words_known"],
            "development_stage": result["development_stage"],
        },
    )
    collective._save(name)
    emit("stats_update", node_stats(name))


@socketio.on("teach")
def on_teach(data):
    name = sessions.get(request.sid)
    if not name:
        return
    data = data or {}
    concept = (data.get("concept") or "").strip()
    explanation = (data.get("explanation") or "").strip()
    examples = data.get("examples") or []
    if not concept:
        return

    # Teach locally, then share with the whole collective.
    collective.teach(name, concept, explanation, examples)
    events = collective.share(name, concept)

    emit(
        "teach_result",
        {"success": True, "concept": concept, "shared_with": len(events)},
    )
    emit("stats_update", node_stats(name))

    # Tell every other node's browser what just arrived from the collective.
    for ev in events:
        socketio.emit(
            "collective_learned",
            {"term": ev["term"], "from": ev["from"], "definition": ev["definition"]},
            room=ev["to"],
        )
        socketio.emit("stats_update", node_stats(ev["to"]), room=ev["to"])

    broadcast_collective()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
