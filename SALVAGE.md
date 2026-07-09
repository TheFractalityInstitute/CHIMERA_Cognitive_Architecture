# CHIMERA — Salvage Report & Roadmap

*Assessment and rescue performed July 2026.*

## TL;DR

The repo was **not junk, but it did not run**. It was a scrapbook of several
parallel experiments (three monolithic `chimera-v0X` scripts plus a modular
rewrite) that were merged without ever being reconciled. The package was named
`chimera_core/` but ~20 files imported from a `chimera.` package that doesn't
exist; the package's `__init__.py` eagerly imported modules that were never
created; and the one genuinely complete subsystem — the organic
language-learning engine — had a method it called but never defined, proving it
had never actually been executed.

**What changed in this pass:** the language-learning core was wired to the web
UI into a small vertical slice that actually boots, learns, persists, and is
covered by tests. Nothing was deleted — the dormant subsystems remain in the
tree for staged revival.

## What works now ✅

| Piece | File | Status |
|---|---|---|
| Organic language-learning engine | `chimera_core/language/chimera_language_learning.py` | **Working, tested** |
| Web chat server (Flask + Socket.IO) | `web/app.py` | **Working** — rewritten around the engine |
| Web UI | `web/templates/index.html` | **Working** — untouched; contract matches |
| Entry point | `run.py` | **Working** |
| Device embodiment (felt state) | `chimera_core/sensors/embodiment.py` | **Working, tested** |
| Native phone sensors (Termux) | `chimera_core/sensors/termux_sensors.py` | **Working, tested** |
| Cloud brain (Neo4j graph) | `chimera_core/collective/graph_brain.py` | **Built + self-test; wiring next** |
| Family web UI (multi-node collective) | `web/app.py`, `web/templates/index.html` | **Working** |
| In-process collective coordinator | `chimera_core/collective/local.py` | **Working, tested** |
| Collective hub (shared knowledge) | `chimera_core/collective/hub.py` | **Working, tested** |
| Collective node client | `chimera_core/collective/client.py` | **Working, tested** |
| Collective demo | `scripts/collective_demo.py` | **Working** — one command |
| Tests | `tests/` (smoke, collective, local, embodiment) | **19 passing** |
| Packaging | `pyproject.toml` | `pip install -e .` |

Run the family UI:  `pip install -e . && python run.py` → http://localhost:5000
Run the collective demo:  `python scripts/collective_demo.py`
Run the networked hub:  `python -m chimera_core.collective.hub` → http://localhost:5001

### The collective is compatible with the single node — and now built on it

An earlier version of this document framed "single node" and "collective" as two
competing products. That was wrong about the *concept*: the intended architecture
(each device runs its own node; nodes pool their learning into a collective) is
coherent and layered, and the collective is now implemented **on top of** the
working node. What was true is that the *old* collective code
(`collective/server.py`, `collective/mobile_client.py`) targeted a different,
never-built "embodied/quantum" agent interface (`chimera.energy`,
`chimera.consciousness_state`, `chimera.canon`) and had missing imports — so it
was rebuilt fresh against the real `OrganicLearningSystem` rather than bridged.

## What was broken (and how it was fixed)

1. **`chimera_core/__init__.py` poisoned every import** — it eagerly imported
   `sensors.chimera_complete`, `cognition.council`, `core.clock`,
   `agents.sensory`, etc., none of which exist at those paths. → Rewritten to a
   lightweight package init that imports nothing heavy.
2. **`web/app.py` targeted a non-existent `chimera.` package** and a full
   `MemoryManager`/DB/crystallization stack that isn't wired up, and mixed
   `async def` handlers with Socket.IO's threading mode (coroutines never
   awaited). → Rewritten to talk directly to `OrganicLearningSystem`, with a
   single long-lived asyncio loop on a background thread and synchronous
   handlers. Persistence uses the engine's own JSON `save_state`/`load_state`.
3. **`OrganicLanguageProcessor._update_from_exchange` was called but never
   defined** — an `AttributeError` on the very first message. → Implemented in
   the author's existing idiom (records the exchange, updates moving-average
   self-assessment). *This is the smoking gun that the core had never run.*
4. **No `.gitignore`, no packaging, no tests, misleading README.** → Added all
   four; README now describes reality.

## What is dormant (kept for revival, not deleted) 💤

These import the phantom `chimera.` package or depend on unbuilt pieces. They
are real ideas worth reviving **one subsystem at a time**, each behind its own
tests:

- `chimera_core/core/council.py` + `eidolon_modules/` — the "Council of Six"
  phase-locked multi-agent orchestrator. Imports `chimera.eidolon_modules.*`
  (wrong package path). Closest to revivable.
- `chimera_core/memory/` (`manager.py`, `persistence.py`, `cache.py`,
  `vector_store.py`) — SQLite + cache + vector store. Imports `chimera.memory.*`;
  `MemoryManager.__init__` also spawns asyncio tasks with no running loop.
- `chimera_core/collective/server.py`, `mobile_client.py`, `mesh_network.py`,
  `hybrid_architecture.py` + `scripts/distributed_mesh.py` + `server/` — the
  **old** "embodied/quantum" collective + P2P mesh. Superseded by the new
  `hub.py`/`client.py`; kept as reference for ideas (phase-locking, mesh
  routing, energy-aware coordination) not yet ported.
- `chimera_core/sensors/`, `integration/`, `interface/`, `ui/`,
  `mobile/` — phone/Garmin biometrics and mobile app. Hardware-dependent.
- `docs/NEED_SORTED/chimera-v05/06/07*.py` — the original monoliths. **Reference
  only**; superseded by the modular tree. Do not import.

## Known rough edges in the working core

- Response generation is deliberately primitive (single-word curiosity prompts).
  That's by design — this is organic acquisition, not a pretrained model.
- `teach()` was made *authoritative* (July 2026): teaching a word now records its
  meaning and marks it `taught` even if the word was already auto-discovered
  (e.g. because it appears in its own examples). Previously the taught definition
  could be silently dropped — which showed up as empty definitions propagating
  through the collective. Vocabulary keys are normalized to lowercase.

## Suggested roadmap

1. **Solidify the core** (done): boots, learns, persists, tested. ✅
2. **Working collective** (done): nodes pool learned concepts through a hub,
   with a demo, a live dashboard, and tests. ✅
3. **Family web UI** (done): one app, each browser names its own CHIMERA, and
   teaching a word propagates live to everyone else's node with attribution. ✅
4. **Device embodiment** (done): browser *and* native-phone (Termux) sensors
   stream motion/battery/light into a felt body-state; CHIMERA reacts and grounds
   taught words in what it was sensing. Each phone hosts its own CHIMERA and reads
   its own hardware (`sensors/termux_sensors.py`, `docs/TERMUX_SETUP.md`). ✅
   Next for embodiment: more sensors (gyroscope, proximity, step counter,
   location), and feed sensations into `interact()` so a shake can drive a full
   cognitive response (the hook already exists).
5. **Cloud collective / shared neural net** (in progress) — chosen model:
   *individuals + a collective mind above them*. `graph_brain.py` implements the
   Neo4j schema (Mind/Concept/Episode, PART_OF/KNOWS/EXPERIENCED/ABOUT) with a
   self-test; `docs/NEO4J_SETUP.md` walks through a free Aura instance. **Next:**
   once the self-test passes on a real instance, wire it into `web/app.py` so
   nodes persist concepts + episodes and restore themselves from the cloud
   (persistence of memory + continuity of experience).
6. **Revive memory persistence** — fix `chimera.memory.*` → `chimera_core.memory.*`,
   remove the loop-spawning from `MemoryManager.__init__`, add tests, then let
   nodes persist thoughts to SQLite instead of a single JSON blob, and give the
   hub durable shared memory.
5. **Revive the Council** — repoint its imports, replace the placeholder
   Language/Interoceptive eidolons, expose it as an *optional* deeper reasoning
   path behind the chat.
6. Port the good ideas from the legacy collective (phase-locking, mesh routing,
   energy-aware coordination) and consider sensors / mobile, each as its own
   tested module.

## How to verify at any point

```bash
pip install -e ".[dev]"
pytest                            # 9 tests (core + collective) must stay green
python run.py                     # http://localhost:5000 — chat + teach
python scripts/collective_demo.py # watch two nodes pool their knowledge
python -m chimera_core.collective.hub  # live collective dashboard on :5001
```
