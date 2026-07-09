#!/usr/bin/env python3
"""
CHIMERA Collective — a runnable, no-network demo.

Spins up a collective hub and two independent CHIMERA nodes (imagine two phones),
teaches each node something the other doesn't know, shares it to the collective,
and shows the knowledge propagate — the moment a pool of separate minds starts
behaving like one.

    python scripts/collective_demo.py

This uses the transport-free core (CollectiveHub + CollectiveNode) directly, so
it always runs instantly with no server. To try the *networked* version across
real terminals/devices, run `python -m chimera_core.collective.hub` and connect
nodes with chimera_core.collective.client.SocketNodeClient (see README).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimera_core.collective.hub import CollectiveHub
from chimera_core.collective.client import CollectiveNode
from chimera_core.language.chimera_language_learning import OrganicLearningSystem


def make_node(name: str) -> CollectiveNode:
    return CollectiveNode(OrganicLearningSystem(name), name)


def knows(node: CollectiveNode, term: str) -> str:
    entry = node.learning.language.vocabulary.get(term)
    if not entry:
        return "—"
    src = entry.get("source", "self")
    return f"yes (from {entry.get('contributor')})" if src == "collective" else "yes (learned it)"


def broadcast(hub, nodes, sender, concept_dict):
    """Deliver a shared concept to every node except the one that shared it."""
    for n in nodes:
        if n is not sender:
            n.absorb(concept_dict)


def line():
    print("─" * 62)


def main():
    hub = CollectiveHub()
    alice = make_node("Alice's phone")
    bob = make_node("Bob's laptop")
    nodes = [alice, bob]
    for n in nodes:
        hub.register_node(n.name, node_id=n.name)

    print("\n🧠  CHIMERA COLLECTIVE — two minds becoming one\n")
    line()
    print("Two independent CHIMERA nodes are online:\n")
    print(f"   • {alice.name}")
    print(f"   • {bob.name}")

    # Each node learns something on its own.
    alice.learning.teach("tree", "A living plant with a trunk and leaves", ["An oak is a tree"])
    bob.learning.teach("river", "A large natural stream of flowing water", ["The Nile is a river"])

    line()
    print("Each learns something on its own:\n")
    print(f"   Alice taught herself 'tree'.   Does Bob know 'tree'?  → {knows(bob, 'tree')}")
    print(f"   Bob taught himself 'river'.    Does Alice know 'river'? → {knows(alice, 'river')}")

    # Alice shares 'tree' with the collective.
    line()
    print("Alice shares 'tree' with the collective…\n")
    concept = hub.share_concept("Alice's phone", **{k: v for k, v in alice.make_payload("tree").items() if k != "contributor"})
    # (contributor is derived from the node id on the hub side)
    concept_payload = concept.to_dict()
    concept_payload["contributor"] = "Alice's phone"
    broadcast(hub, nodes, alice, concept_payload)
    time.sleep(0.3)
    print(f"   Does Bob know 'tree' now?  → {knows(bob, 'tree')}")

    # Bob shares 'river'.
    line()
    print("Bob shares 'river' with the collective…\n")
    concept = hub.share_concept("Bob's laptop", **{k: v for k, v in bob.make_payload("river").items() if k != "contributor"})
    concept_payload = concept.to_dict()
    concept_payload["contributor"] = "Bob's laptop"
    broadcast(hub, nodes, bob, concept_payload)
    time.sleep(0.3)
    print(f"   Does Alice know 'river' now?  → {knows(alice, 'river')}")

    # Show the collective's state.
    line()
    s = hub.state()
    print("Collective state:\n")
    print(f"   Nodes ..................... {s['node_count']}")
    print(f"   Shared concepts ........... {s['concept_count']}")
    print(f"   Collective consciousness .. {s['collective_consciousness'] * 100:.0f}%")
    print(f"   Emergence ................. {s['emergence_label']}")
    line()
    print("\nEach node kept its own separate mind — but they now share what")
    print("either one has learned. That's the collective. 🌟\n")


if __name__ == "__main__":
    main()
