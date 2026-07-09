"""
CHIMERA web application (minimal working core).

This is the honest, runnable heart of CHIMERA: a web front-end wired directly to
the OrganicLearningSystem, which learns language and forms thoughts through
conversation. State persists to a JSON file between runs.

Deliberately NOT wired in yet: the distributed "collective" server, the sensor /
biometric subsystems, the SQLite MemoryManager, and the crystallization/curiosity
agents. Those exist in the tree but were never integrated; see SALVAGE.md for the
roadmap to fold them back in one at a time.
"""

import asyncio
import atexit
import threading
import time
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

import sys

# Make the repo root importable whether run via `python run.py` or `python web/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimera_core.language.chimera_language_learning import OrganicLearningSystem

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = Flask(__name__)
app.config["SECRET_KEY"] = "chimera-secret-key-change-in-production"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "chimera_state.json"

# --------------------------------------------------------------------------- #
# A dedicated asyncio loop, running on a background thread.
#
# OrganicLearningSystem.interact() is a coroutine, but Flask-SocketIO runs its
# handlers synchronously in `threading` mode. Rather than spin up a new event
# loop per message (slow, and it discards state held on the loop), we keep one
# long-lived loop and submit coroutines to it thread-safely.
# --------------------------------------------------------------------------- #

_loop = asyncio.new_event_loop()


def _run_loop() -> None:
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop, daemon=True).start()


def run_async(coro):
    """Run a coroutine on the background loop and block for its result."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# --------------------------------------------------------------------------- #
# CHIMERA state
# --------------------------------------------------------------------------- #

learning = OrganicLearningSystem("chimera_web")
active_sessions: dict[str, dict] = {}
_milestones_sent = 0  # index into learning.development_log already broadcast
_state_lock = threading.Lock()


def _load_state() -> None:
    if STATE_FILE.exists():
        try:
            learning.load_state(str(STATE_FILE))
            print(f"✓ Restored learning state from {STATE_FILE}")
        except Exception as exc:  # corrupt/old snapshot shouldn't block startup
            print(f"! Could not load prior state ({exc}); starting fresh")


def _save_state() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _state_lock:
        learning.save_state(str(STATE_FILE))
    return str(STATE_FILE)


_load_state()
atexit.register(_save_state)


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    return jsonify(
        {
            "online": True,
            "thoughts": len(learning.reasoning.thought_network),
            "concepts": len(learning.language.vocabulary),
            "conversations": learning.conversation_count,
            "curiosity_level": learning.curiosity_level,
            "development_stage": learning._get_development_stage(),
            "active_sessions": len(active_sessions),
        }
    )


@app.route("/api/thoughts")
def get_thoughts():
    thoughts = [
        {
            "id": tid,
            "type": "thought",
            "content": t.symbolic_form,
            "confidence": t.confidence,
            "connections": len(t.connections),
            "timestamp": t.timestamp,
        }
        for tid, t in learning.reasoning.thought_network.items()
    ]
    thoughts.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(thoughts[:100])


@app.route("/api/concepts")
def get_concepts():
    concepts = [
        {
            "term": word,
            "confidence": data.get("confidence", 0),
            "count": data.get("count", 0),
            "meanings": data.get("meanings", []),
            "taught": data.get("taught", False),
        }
        for word, data in learning.language.vocabulary.items()
    ]
    concepts.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(concepts)


@app.route("/api/memory/save", methods=["POST"])
def save_memory():
    try:
        path = _save_state()
        return jsonify({"success": True, "path": path})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/export/json")
def export_json():
    data = {
        "chimera_id": learning.chimera_id,
        "vocabulary": learning.language.vocabulary,
        "thoughts": {
            tid: {
                "symbolic_form": t.symbolic_form,
                "confidence": t.confidence,
                "connections": list(t.connections),
            }
            for tid, t in learning.reasoning.thought_network.items()
        },
        "conversation_count": learning.conversation_count,
        "development_log": learning.development_log,
    }
    return jsonify(data)


# --------------------------------------------------------------------------- #
# Curiosity (lightweight, derived from current vocabulary)
# --------------------------------------------------------------------------- #


def _make_curiosity() -> dict:
    vocab = learning.language.vocabulary
    if vocab:
        word = min(vocab, key=lambda w: vocab[w].get("confidence", 0))
        return {
            "question": f"I keep seeing '{word}' but I'm not sure I understand it. What does it mean?",
            "target": word,
            "priority": round(1.0 - vocab[word].get("confidence", 0), 2),
        }
    return {
        "question": "I don't know any words yet. Could you teach me one?",
        "target": None,
        "priority": 0.5,
    }


# --------------------------------------------------------------------------- #
# WebSocket handlers
# --------------------------------------------------------------------------- #


@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    active_sessions[session_id] = {"connected_at": time.time(), "messages": 0}
    join_room(session_id)
    emit("connected", {"session_id": session_id, "message": "Connected to CHIMERA"})


@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    active_sessions.pop(session_id, None)
    leave_room(session_id)


def _broadcast_new_milestones():
    global _milestones_sent
    log = learning.development_log
    while _milestones_sent < len(log):
        socketio.emit("milestone", {"text": log[_milestones_sent], "timestamp": time.time()})
        _milestones_sent += 1


def _broadcast_stats():
    socketio.emit(
        "stats_update",
        {
            "total_thoughts": len(learning.reasoning.thought_network),
            "total_concepts": len(learning.language.vocabulary),
            "total_conversations": learning.conversation_count,
            "abstraction_level": learning.abstraction_level,
            "active_users": len(active_sessions),
        },
    )


@socketio.on("message")
def handle_message(data):
    session_id = request.sid
    text = (data or {}).get("text", "").strip()
    if not text:
        return

    if session_id in active_sessions:
        active_sessions[session_id]["messages"] += 1

    result = run_async(learning.interact(text))

    emit(
        "response",
        {
            "text": result["response"],
            "confidence": result["understanding"],
            "thoughts_formed": result["thoughts_formed"],
            "words_known": result["words_known"],
            "development_stage": result["development_stage"],
            "curiosity": result["curiosity"],
        },
    )

    _broadcast_new_milestones()
    _broadcast_stats()


@socketio.on("teach")
def handle_teach(data):
    data = data or {}
    concept = data.get("concept", "").strip()
    explanation = data.get("explanation", "").strip()
    examples = data.get("examples", []) or []
    if not concept:
        return

    result = learning.teach(concept, explanation, examples)
    _save_state()

    emit(
        "teach_result",
        {
            "success": True,
            "concept": concept,
            "understanding": result["current_understanding"],
        },
    )
    _broadcast_stats()


@socketio.on("request_curiosity")
def handle_curiosity_request():
    emit("curiosity", _make_curiosity(), room=request.sid)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
