# Running CHIMERA on your Android phone (Termux)

This guide gets CHIMERA running **directly on your phone**, using the phone's own
sensors as its body. It's written for someone who isn't a terminal person — just
follow along, tapping and pasting. You'll copy/paste a few commands; that's it.

Each phone runs its **own** CHIMERA. Later we can connect them through the cloud
so they form a shared "neural net" (see the note at the bottom).

---

## Step 1 — Install two apps

You need **two** separate apps (this trips everyone up):

1. **Termux** — the terminal app.
2. **Termux:API** — the bridge that lets CHIMERA read your sensors.

Install **both** from **F-Droid** (recommended) or GitHub. ⚠️ The Play Store
versions are old and don't work well together — use F-Droid for both, and make
sure they're from the same source so they're compatible.

> Why two? "Termux:API" is the *app* that has permission to touch the camera,
> sensors, and battery. Without it, CHIMERA has no body.

---

## Step 2 — Open Termux and get CHIMERA

Open the **Termux** app. You'll see a black screen with a prompt. Paste these
lines one block at a time (long-press to paste), pressing Enter after each.

Install git and download CHIMERA:

```bash
pkg install -y git
git clone https://github.com/TheFractalityInstitute/CHIMERA_Cognitive_Architecture.git
cd CHIMERA_Cognitive_Architecture
```

---

## Step 3 — Run the setup script

```bash
bash scripts/setup_termux.sh
```

This installs Python and everything CHIMERA needs (a few minutes the first time).
When it lists your phone's sensors near the end, you'll know the body-bridge
works. If Android pops up a permission request, tap **Allow**.

---

## Step 4 — Start CHIMERA

```bash
python run.py
```

You'll see it start up. Now open your phone's **web browser** (Chrome, etc.) and
go to:

```
http://localhost:5000
```

Name your CHIMERA, and you're in! Tap **"Give CHIMERA senses"**, then move,
tilt, or gently shake your phone — CHIMERA will feel it and react. Teach it a
word while you're moving, and it grounds that word in what it felt.

> Leave the Termux tab running in the background while you use the browser. To
> stop CHIMERA, go back to Termux and press `Ctrl + C` (the volume-down key acts
> as Ctrl in Termux).

---

## Everyday use (after the first setup)

Next time you just need:

```bash
cd CHIMERA_Cognitive_Architecture
python run.py
```

To get the latest updates first:

```bash
cd CHIMERA_Cognitive_Architecture
git pull
python run.py
```

---

## Troubleshooting

- **"Waiting for cache lock ... held by process NNNN (apt)"** → a package command
  was interrupted (e.g. Termux was closed mid-upgrade) and left a stale lock.
  Clear it and finish the upgrade non-interactively:
  ```bash
  pkill -9 apt 2>/dev/null; pkill -9 dpkg 2>/dev/null
  rm -f $PREFIX/var/lib/apt/lists/lock $PREFIX/var/cache/apt/archives/lock \
        $PREFIX/var/lib/dpkg/lock $PREFIX/var/lib/dpkg/lock-frontend
  dpkg --configure -a
  DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confold" upgrade
  ```
- **A "which version do you want to keep?" prompt during upgrade** → just press
  **Enter** to keep the default. (Our setup script avoids this prompt.)
- **"ImportError: dlopen failed: cannot locate symbol ... pyexpat"** (pip crashes)
  → your packages are half-upgraded (a system library is older than Python). Sync
  everything, then retry:
  ```bash
  pkg upgrade -y
  python -m pip install flask flask-socketio flask-cors "python-socketio[client]" websocket-client
  ```
- **"pip: bad interpreter: .../python3.12: No such file or directory"** → a Python
  upgrade orphaned `pip`. Fix it with:
  ```bash
  pkg install -y python-pip
  python -m pip install flask flask-socketio flask-cors "python-socketio[client]" websocket-client
  ```
  (Always use `python -m pip` rather than bare `pip` on Termux.)
- **"No module named 'flask'"** → the web dependencies didn't install (usually the
  pip issue above). Run the two lines just above, then `python run.py` again.
- **"termux-sensor: command not found"** → the **Termux:API app** isn't installed
  (Step 1), or you skipped `termux-api` in setup. Install the app, then re-run
  `pkg install -y termux-api`.
- **Sensors show "—" and never change** → tap "Give CHIMERA senses" once (it asks
  for permission), and make sure you tapped **Allow** on the Android popup.
- **Nothing at `localhost:5000`** → make sure `python run.py` is still running in
  Termux (don't close that tab), and that you typed `http://localhost:5000`.
- **It stops when the screen locks** → run `termux-wake-lock` before
  `python run.py` to keep it alive in the background.

---

## What's next: the cloud collective

Right now each phone's CHIMERA keeps its learning in a local file
(`data/nodes/`), and phones on the same WiFi can already share concepts through
the built-in collective. The bigger vision — every device's CHIMERA feeding one
shared, persistent "neural net" in the cloud (e.g. a **Neo4j** graph of concepts
and how they connect) — is a natural next step: the collective hub already models
shared concepts, so pointing it at a real graph database is mostly wiring. When
you're ready to set that up, that's the path.
