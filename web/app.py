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
from chimera_core.sensors.embodiment import EmbodiedSenses
from chimera_core.sensors import termux_sensors
from chimera_core.collective.graph_brain import GraphBrain
from chimera_core.language.local_llm import LocalLLM

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
senses_by_name: dict[str, EmbodiedSenses] = {}     # node name -> its body-state

# The cloud brain (Neo4j) — optional. If credentials are present it becomes the
# durable, shared memory; if not, everything runs locally exactly as before.
brain = GraphBrain.from_env()
BRAIN_ON = False
mind_ids: dict[str, str] = {}                      # node name -> graph mind id


def _init_brain():
    global BRAIN_ON
    if not brain.available():
        print("· No cloud brain configured — memory is local only (see docs/NEO4J_SETUP.md)")
        return
    try:
        brain.connect()
        BRAIN_ON = True
        print("✓ Cloud brain connected — CHIMERA will persist and remember ☁️")
    except Exception as exc:
        print(f"! Cloud brain not reachable ({exc}); continuing with local memory")


_init_brain()

# The local voice (Ollama) — optional. When a local model is running on this
# device, CHIMERA speaks through it, grounded in its own mind; otherwise it uses
# the simple built-in chat. Nothing leaves the device either way.
local_llm = LocalLLM()
LLM_ON = local_llm.available()
if LLM_ON:
    print(f"✓ Local voice online — CHIMERA will speak through '{local_llm.model}' 🗣️")
else:
    print("· No local model running — using the simple built-in voice (see docs/LOCAL_MODEL_SETUP.md)")

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


# -- cloud brain helpers (all no-ops when BRAIN_ON is False) ------------------ #


def _brain_mind(name: str):
    """Ensure this CHIMERA exists in the cloud graph; return its mind id (or None)."""
    if not BRAIN_ON:
        return None
    if name not in mind_ids:
        try:
            mind_ids[name] = brain.ensure_mind(name)
        except Exception as exc:
            print(f"! cloud ensure_mind failed: {exc}")
            return None
    return mind_ids[name]


def _restore_from_cloud(name: str) -> dict:
    """Pull a CHIMERA's remembered concepts from the cloud into its local mind."""
    mind_id = _brain_mind(name)
    if not mind_id:
        return {"remembered": 0, "experiences": 0}
    try:
        node = collective.get_or_create(name)
        concepts = brain.known_concepts(mind_id)
        for c in concepts:
            node.learning.teach(c["term"], c.get("definition") or "")
            entry = node.learning.language.vocabulary.get(c["term"])
            if entry is not None:
                entry["source"] = "memory"  # restored from the cloud
        collective._save(name)
        experiences = len(brain.timeline(mind_id, limit=1000))
        return {"remembered": len(concepts), "experiences": experiences}
    except Exception as exc:
        print(f"! cloud restore failed: {exc}")
        return {"remembered": 0, "experiences": 0}


def _brain_remember(name: str, term: str, definition: str, felt, teacher):
    """Persist a taught concept + episode to the cloud (in the background)."""
    mind_id = _brain_mind(name)
    if not mind_id:
        return

    def work():
        try:
            brain.remember_concept(mind_id, term, definition, felt=felt, teacher=teacher)
        except Exception as exc:
            print(f"! cloud remember failed: {exc}")

    socketio.start_background_task(work)


def _brain_episode(name: str, kind: str, text: str, felt):
    """Persist a non-concept experience (a sensation) to the cloud in the background."""
    mind_id = _brain_mind(name)
    if not mind_id:
        return

    def work():
        try:
            brain.record_episode(mind_id, kind, text, felt)
        except Exception as exc:
            print(f"! cloud episode failed: {exc}")

    socketio.start_background_task(work)


def emit_cloud_status(target_room=None):
    payload = {"on": BRAIN_ON}
    if BRAIN_ON:
        try:
            payload.update(brain.collective_state())
        except Exception:
            pass
    if target_room:
        socketio.emit("cloud_status", payload, room=target_room)
    else:
        socketio.emit("cloud_status", payload)


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
    # Bring this CHIMERA's memory back from the cloud, if it has one.
    restored = _restore_from_cloud(name)
    emit(
        "joined",
        {
            "name": name,
            "stats": node_stats(name),
            "remembered": restored["remembered"],
            "experiences": restored["experiences"],
        },
    )
    emit_cloud_status(target_room=name)
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
    # Always run the organic engine: it grows the concept graph and stats, and
    # supplies the simple fallback reply.
    result = run_async(node.learning.interact(text))
    reply = result["response"]
    voice = "simple"

    # If a local model is running, let it speak — grounded in this mind's own
    # words, felt state, and recent experiences. Falls back on any failure.
    if LLM_ON:
        body = senses_by_name.get(name)
        grounded = local_llm.respond(
            name,
            text,
            words=list(node.learning.language.vocabulary.keys()),
            feeling=body.feeling() if body else None,
            experiences=node.learning.development_log[-5:],
        )
        if grounded:
            reply = grounded
            voice = "gemma"

    emit(
        "response",
        {
            "text": reply,
            "voice": voice,
            "confidence": result["understanding"],
            "thoughts_formed": result["thoughts_formed"],
            "words_known": result["words_known"],
            "development_stage": result["development_stage"],
        },
    )
    collective._save(name)
    emit("stats_update", node_stats(name))


@socketio.on("senses")
def on_senses(data):
    name = sessions.get(request.sid)
    if not name:
        return
    body = senses_by_name.setdefault(name, EmbodiedSenses())
    result = body.update(data or {})

    emit(
        "sense_state",
        {
            "feeling": result["feeling"],
            "motion": result["motion"],
            "light": result["light"],
            "energy": result["energy"],
        },
    )
    # A notable sensation (picked up, shaken, went dark…) becomes a spoken reaction
    # and a remembered experience in the cloud timeline.
    if result["reaction"]:
        emit("sensation", {"text": result["reaction"]})
        event = result["events"][0] if result["events"] else "sensed"
        _brain_episode(name, "sensed", event, result["feeling"])


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

    # Ground the concept in what CHIMERA was sensing when it learned it.
    felt = None
    body = senses_by_name.get(name)
    if body is not None:
        felt = body.feeling()
        entry = collective.get_or_create(name).learning.language.vocabulary.get(concept.lower())
        if entry is not None:
            entry["felt"] = felt
            collective._save(name)  # persist the grounding (teach() saved before this)

    events = collective.share(name, concept)

    # Persist to the cloud brain — the mind's own memory + the collective's.
    _brain_remember(name, concept, explanation, felt, teacher=name)

    emit(
        "teach_result",
        {"success": True, "concept": concept, "shared_with": len(events)},
    )
    emit("stats_update", node_stats(name))
    emit_cloud_status(target_room=name)

    # Tell every other node's browser what just arrived from the collective.
    for ev in events:
        socketio.emit(
            "collective_learned",
            {"term": ev["term"], "from": ev["from"], "definition": ev["definition"]},
            room=ev["to"],
        )
        socketio.emit("stats_update", node_stats(ev["to"]), room=ev["to"])

    broadcast_collective()


# --------------------------------------------------------------------------- #
# Native device body (Termux) — this phone's real sensors feed its CHIMERA(s)
# --------------------------------------------------------------------------- #


def _apply_device_senses(reading: dict) -> None:
    """A reading from the physical device flows into every active local node."""
    for name, sids in list(name_sids.items()):
        if not sids:
            continue
        body = senses_by_name.setdefault(name, EmbodiedSenses())
        result = body.update(reading)
        socketio.emit(
            "sense_state",
            {k: result[k] for k in ("feeling", "motion", "light", "energy")},
            room=name,
        )
        if result["reaction"]:
            socketio.emit("sensation", {"text": result["reaction"]}, room=name)


def start_device_senses() -> None:
    """If we're running on a phone (Termux), stream its hardware senses in."""
    if termux_sensors.available():
        print("✓ Termux sensors detected — CHIMERA has a body 🖐️")
        socketio.start_background_task(termux_sensors.stream, _apply_device_senses)
    else:
        print("· No Termux sensors here — browser senses still work (see the Senses panel)")


start_device_senses()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
