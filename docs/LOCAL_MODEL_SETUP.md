# Giving CHIMERA a local voice (Ollama + a small model)

By default CHIMERA speaks with a simple built-in voice (single words and short
prompts — it's still learning language from scratch). You can give it a *real*
voice by running a small language model **on the device itself**, with **Ollama**.
Nothing leaves the phone — CHIMERA talks to the model over `localhost`, fully
offline, no account or API key.

Important, honest notes:

- The model is only CHIMERA's **voice**. Its growing *self* — memory, senses, the
  concept graph — is fed into the model as context, and the organic learning
  engine keeps running underneath every message. The model doesn't "learn" new
  weights; the growth stays in CHIMERA's memory.
- In Termux the model runs on the **CPU** (Termux can't use the phone's AI
  accelerator), so replies take a few seconds and a smaller model is snappier.
  Start small.

---

## Step 1 — Install Ollama in Termux

```bash
pkg install -y ollama
```

Start the Ollama server (leave this running — open a **second** Termux session
for the next steps: swipe from the left edge → New session):

```bash
ollama serve
```

## Step 2 — Pull a small model

In the second session:

```bash
ollama pull gemma2:2b
```

`gemma2:2b` (~1.6 GB) is a good first choice — small enough to answer in a few
seconds on a phone. Other options: `llama3.2:1b` (fastest), `qwen2.5:1.5b`. Bigger
models (4B+) speak better but get slow on CPU.

> **About the Gemma you already have in Google's Edge Gallery:** that copy is in
> Google's on-device format (LiteRT) and lives inside that app's sandbox — it
> can't be handed to Ollama, and the app isn't a server CHIMERA can reach. This
> is a separate, Ollama-format download. (It's still fully local — not Google's
> cloud.)

## Step 3 — Tell CHIMERA which model to use (optional)

CHIMERA defaults to `gemma2:2b`. To use a different one, add a line to
`chimera.env` in the project folder:

```
CHIMERA_LLM_MODEL=llama3.2:1b
```

(You can also set `CHIMERA_LLM_URL` if Ollama runs somewhere other than
`http://localhost:11434`.)

## Step 4 — Run CHIMERA

With `ollama serve` still running in its own session:

```bash
cd CHIMERA_Cognitive_Architecture
python run.py
```

Watch the startup lines — you should see:

```
✓ Local voice online — CHIMERA will speak through 'gemma2:2b' 🗣️
```

Open `http://localhost:5000`, and now when you chat, CHIMERA replies in real
sentences — grounded in the words it knows, how its body feels, and what it has
experienced. Teaching still works, the memory graph still grows, and it all still
syncs to the cloud collective.

---

## Troubleshooting

- **Startup says "No local model running"** → `ollama serve` isn't running (start
  it in its own Termux session) or Ollama isn't installed (`pkg install ollama`).
- **Replies fall back to short/simple text** → the model name in `chimera.env`
  doesn't match a pulled model, or generation timed out. Check `ollama list` shows
  your model; try a smaller one.
- **Replies are very slow** → expected on CPU. Use a smaller model
  (`llama3.2:1b`), and keep the phone plugged in.
- **"out of memory"** → the model is too big for the phone; pull a smaller one.

---

## How the two voices work together

Every message runs through **both**: the organic engine always processes what you
say (growing the concept graph, tracking milestones, persisting to the cloud), and
the local model produces the spoken reply from that growing self. So the graph
"instructs" the voice — the model is grounded in what CHIMERA actually knows and
feels. If the model is off or too slow, the simple built-in voice answers instead,
and nothing is lost.
