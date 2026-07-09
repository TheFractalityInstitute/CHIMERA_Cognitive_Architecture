# CHIMERA Cognitive Architecture

**An experimental multi-agent cognitive architecture that develops language and
understanding organically through conversation.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

> **Status (2026):** This repository was assembled from several earlier
> experiments and, for a long time, did not run as an integrated whole. It has
> now been given a working, tested **core**: a web chat interface backed by an
> organic language-learning system, plus a working **collective** that pools
> multiple nodes into shared intelligence. Some modules in the tree
> (phone/biometric sensors, mobile UI, the older "embodied/quantum" collective
> code) are **not yet integrated** and are kept as reference for future work.
> See [`SALVAGE.md`](SALVAGE.md) for exactly what works, what doesn't, and the
> roadmap.

## What actually works today

- **`python run.py`** launches a friendly web app at `http://localhost:5000`.
  Open it in a browser, name your CHIMERA, and start chatting and teaching — no
  terminal needed to use it.
- It's a **family collective**: everyone can name their own CHIMERA (in another
  tab or on another device on the same WiFi), and a word taught to one shows up
  on the others live — "🌐 The collective taught me 'dragon' (from Dante's
  CHIMERA)!"
- Each CHIMERA forms "thoughts," discovers words, tracks developmental
  milestones, and reports how much it understood.
- Every CHIMERA's learning **persists** to `data/nodes/` between runs.
- A **test suite** (`tests/`) covers the language core and the collective.

## Quick start

```bash
# 1. Install (editable, so the chimera_core package resolves everywhere)
pip install -e .

# 2. Run
python run.py
```

Then open **http://localhost:5000** in your browser, give your CHIMERA a name,
and you're in. To play as a family, have each person open the same address —
either in another browser tab, or from another device on your home WiFi using
`http://<the-host-computer's-IP>:5000`. Everyone picks their own name.

**Teach a word** with the *Teach CHIMERA a word* box (word + what it means), and
watch it appear on everyone else's CHIMERA in the Collective panel. Just chatting
also grows its vocabulary over time.

```bash
# (optional) run the tests
pip install -e ".[dev]"
pytest
```

## The collective — many minds, one shared memory

Each person runs their **own** CHIMERA. When one learns a word, it's shared to
the collective and every other CHIMERA absorbs it — so knowledge learned by one
shows up for all, attributed to whoever taught it. The **collective consciousness
%** rises as more minds pool more knowledge (it only counts when 2+ minds are
actually sharing — one mind alone isn't a collective).

The web app runs the whole collective in one process, so the family experience
needs nothing extra. There's also a **standalone demo** and a **networked hub**
for going cross-machine:

```bash
python scripts/collective_demo.py        # see two minds pool knowledge (no server)
python -m chimera_core.collective.hub     # a networked hub + dashboard on :5001
```

The collective *logic* (`chimera_core/collective/hub.py`, `client.py`,
`local.py`) is transport-free and unit-tested; the Socket.IO layers are thin
wrappers over it.

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
  collective/hub.py  client.py  local.py  # ← the working collective (+ legacy refs)
  memory/        core/        eidolon_modules/    # partially-integrated modules
  sensors/     ui/    integration/          # reference / not yet wired
web/
  app.py                                  # ← the working web server
  templates/index.html                    # ← the UI
scripts/collective_demo.py                # ← one-command collective demo
tests/test_smoke.py  tests/test_collective.py   # ← passing tests
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
