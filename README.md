# CHIMERA Cognitive Architecture

**An experimental multi-agent cognitive architecture that develops language and
understanding organically through conversation.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

> **Status (2026):** This repository was assembled from several earlier
> experiments and, for a long time, did not run as an integrated whole. It has
> now been given a working, tested **core**: a web chat interface backed by an
> organic language-learning system. Many other modules in the tree (distributed
> "collective" networking, phone/biometric sensors, mobile UI) are **not yet
> integrated** and are kept as reference for future work. See
> [`SALVAGE.md`](SALVAGE.md) for exactly what works, what doesn't, and the
> roadmap.

## What actually works today

- **`python run.py`** launches a Flask + Socket.IO web app at
  `http://localhost:5000`.
- You can **chat** with CHIMERA and **teach** it concepts; it forms "thoughts,"
  discovers words, tracks developmental milestones, and reports a confidence
  level for how much it understood.
- Learning state **persists** to `data/chimera_state.json` between runs.
- A **smoke test suite** (`tests/test_smoke.py`) proves the core imports, holds
  a conversation, learns a taught concept, and round-trips its saved state.

## Quick start

```bash
# 1. Install (editable, so the chimera_core package resolves everywhere)
pip install -e .

# 2. Run
python run.py
# → open http://localhost:5000

# 3. (optional) run the tests
pip install -e ".[dev]"
pytest
```

### First conversation

```
CHIMERA: Connected to CHIMERA
You:     teach: tree | A living plant with a trunk and leaves | Oak trees are tall
You:     what is a tree?
CHIMERA: ... [confidence shown as a percentage]
```

Use the **Teach** button in the UI (or the `teach:` panel) to add concepts
explicitly; just chatting also grows its vocabulary over time.

## How the core works

When you send text, the `OrganicLanguageProcessor` segments it, discovers or
reinforces words, and constructs a rough "meaning." The `ReasoningEngine` turns
each utterance into a `Thought` — a node with a symbolic form, a confidence, and
connections to similar thoughts. When enough similar thoughts accumulate, an
`AbstractionLayer` forms automatically. The `OrganicLearningSystem` ties these
together, tracks conversation count / vocabulary / abstraction level, and emits
developmental milestones ("First word learned!", "First abstract concept
formed!") as they naturally occur.

There is **no pre-programmed grammar and no training corpus** — understanding is
meant to emerge from repeated interaction. It is a research toy for exploring
organic language acquisition, not a chatbot that already knows things.

## Project layout

```
chimera_core/
  language/chimera_language_learning.py   # ← the working brain (self-contained)
  memory/        core/        eidolon_modules/    # partially-integrated modules
  collective/    sensors/     ui/    integration/  # reference / not yet wired
web/
  app.py                                  # ← the working web server
  templates/index.html                    # ← the UI
tests/test_smoke.py                       # ← core smoke tests
docs/                                     # design notes; NEED_SORTED/ = raw archive
SALVAGE.md                                # status + roadmap for reviving the rest
```

## Ethics

CHIMERA is intended to embody the Fractality Charter of Universal Ethics —
reciprocity, integrity, agency, and consideration of consequence. See
`chimera_core/ethics/fractality_charter.py`.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Acknowledgments

- Created by Grazi ([@GraziTheMan](https://github.com/GraziTheMan))
- Part of the [Fractality Framework](https://github.com/TheFractalityInstitute)

---

*"We are not building a simulation of intelligence — we are cultivating genuine
understanding through interaction, one thought at a time."*
