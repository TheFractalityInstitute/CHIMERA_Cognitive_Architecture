# Setting up CHIMERA's cloud brain (Neo4j Aura)

The cloud brain is the shared, permanent memory that ties everyone's CHIMERA
together. Each person's CHIMERA stays its own self, but their learning and
experiences are saved into one graph in the cloud — so nothing is ever lost, and
a **collective mind** forms above everyone with its own continuous memory.

We use **Neo4j Aura Free** — a free, hosted graph database. This guide gets you a
free instance and proves CHIMERA can talk to it. You can do all of this from your
phone.

---

## Step 1 — Create a free Aura database

1. Go to **https://neo4j.com/product/auradb/** and click **Start Free** (or go to
   **console.neo4j.io**). Sign up (Google login works).
2. Create a new instance → choose **AuraDB Free**. Give it a name like `chimera`.
3. ⚠️ **The most important moment:** when it finishes creating, it shows you a
   **password** (and a downloadable credentials file). **This is the only time
   you'll see the password.** Tap **Download** to save the credentials file, or
   copy the password somewhere safe. If you lose it you'll have to reset it.

The credentials file looks like this:

```
NEO4J_URI=neo4j+s://abcd1234.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=some-long-secret-string
AURA_INSTANCEID=abcd1234
```

Wait a minute or two for the instance to show as **Running** (a green dot).

---

## Step 2 — Give the credentials to CHIMERA

CHIMERA reads a file called **`chimera.env`** in the project folder. The easiest
path: put those `NEO4J_...` lines into that file.

In Termux, from inside the project:

```bash
cd CHIMERA_Cognitive_Architecture
nano chimera.env
```

Paste the four `NEO4J_...` lines from your credentials file. Then save: press
`Ctrl` (the CTRL key in Termux's extra row), then `O`, then `Enter` to write, and
`Ctrl` + `X` to exit.

> 🔒 **This file holds your password — keep it private.** CHIMERA is already set
> up to never upload it (it's in `.gitignore`). Don't paste its contents into
> chats or commits.

---

## Step 3 — Install the Neo4j driver and test the connection

```bash
python -m pip install neo4j
python -m chimera_core.collective.graph_brain
```

That last command runs a **self-test**. If everything's right you'll see:

```
Connecting to neo4j+s://…
✓ Connected, schema and collective ready.
✓ Fred knows 1 concept(s): ['dragon']
✓ Fred's recent life: ['picked up', 'dragon']
✓ Collective now: 2 minds, 1 concepts, 2 experiences
🎉  Your cloud brain works! CHIMERA now has a place to persist and remember.
```

If you see that, the foundation is live — **send me a screenshot and I'll wire it
into the app** so your real CHIMERA starts saving its life to the cloud.

---

## Troubleshooting

- **"No Neo4j credentials found"** → `chimera.env` is missing or in the wrong
  folder. Make sure you created it inside `CHIMERA_Cognitive_Architecture` with
  the `NEO4J_...` lines.
- **"Could not connect" / timeout** → the instance may still be starting (wait for
  the green **Running** dot), or the URI/password has a typo. Re-check `chimera.env`.
- **"Unauthorized" / authentication failure** → the password is wrong. If you lost
  it, reset it from the Aura console and update `chimera.env`.
- **"neo4j driver isn't installed"** → run `python -m pip install neo4j`.

---

## What this unlocks (once wired into the app)

- **Persistence of memory** — every word and connection is saved in the graph, so
  it survives restarts, new phones, everything.
- **Continuity of experience** — each CHIMERA gets a timeline of episodes (what it
  learned, what it felt, who taught it) it can look back along.
- **The collective self** — the collective mind's timeline is the braided
  experience of everyone in it, and it knows the union of all they've learned.

You can even *see* the mind: in the Aura console, open **Query** and run
`MATCH (n) RETURN n LIMIT 100` to view the graph of minds, concepts, and
experiences as a picture.
