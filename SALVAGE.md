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
| Smoke tests | `tests/test_smoke.py` | **4 passing** |
| Packaging | `pyproject.toml` | `pip install -e .` |

Run it: `pip install -e . && python run.py` → http://localhost:5000

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
- `chimera_core/collective/` + `scripts/distributed_mesh.py` + `server/` — the
  distributed "collective consciousness" networking. A separate product vision.
- `chimera_core/sensors/`, `integration/`, `interface/`, `ui/`,
  `mobile/` — phone/Garmin biometrics and mobile app. Hardware-dependent.
- `docs/NEED_SORTED/chimera-v05/06/07*.py` — the original monoliths. **Reference
  only**; superseded by the modular tree. Do not import.

## Known rough edges in the working core

- `OrganicLearningSystem.teach()` only writes a `taught: True` vocabulary entry
  when the word is *new*, but `process_utterance` (run first) has usually already
  inserted it — so taught concepts can stay `taught: False`. Left as-is
  intentionally: it's a learning-semantics decision for the author, not a crash.
- Response generation is deliberately primitive (single-word curiosity prompts).
  That's by design — this is organic acquisition, not a pretrained model.

## Suggested roadmap

1. **Solidify the core** (done): boots, learns, persists, tested. ✅
2. **Revive memory persistence** — fix `chimera.memory.*` → `chimera_core.memory.*`,
   remove the loop-spawning from `MemoryManager.__init__`, add tests, then let
   the web app persist thoughts to SQLite instead of a single JSON blob.
3. **Revive the Council** — repoint its imports, replace the placeholder
   Language/Interoceptive eidolons, and expose it as an *optional* deeper
   reasoning path behind the existing chat.
4. **Pick one big direction.** The repo currently blends two products: (a) this
   single-node organic-learning companion, and (b) a distributed multi-device
   "collective." They pull in different directions — choose (a) *or* (b) as the
   headline and demote the other to a branch/experiment.
5. Only then consider sensors / mobile / mesh, each as its own tested module.

## How to verify at any point

```bash
pip install -e ".[dev]"
pytest                 # core smoke tests must stay green
python run.py          # http://localhost:5000 — chat + teach should work
```
