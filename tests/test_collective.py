"""
Tests for the CHIMERA collective (transport-free core).

These prove the collective's defining behaviour: a concept learned by one node
propagates, through the hub, into another node's own vocabulary — and that the
emergence metric only lights up when there are genuinely multiple nodes sharing
knowledge.
"""

from chimera_core.collective.hub import CollectiveHub
from chimera_core.collective.client import CollectiveNode
from chimera_core.language.chimera_language_learning import OrganicLearningSystem


def _node(name):
    return CollectiveNode(OrganicLearningSystem(name), name)


def test_hub_registers_and_shares():
    hub = CollectiveHub()
    nid = hub.register_node("Alice")
    concept = hub.share_concept(nid, "Tree", "a living plant", ["oak"])

    assert concept is not None
    assert concept.term == "tree"  # normalized to lowercase
    assert "Alice" in concept.contributors
    assert hub.snapshot()[0]["term"] == "tree"


def test_resharing_strengthens_resonance():
    hub = CollectiveHub()
    a = hub.register_node("Alice")
    b = hub.register_node("Bob")

    first = hub.share_concept(a, "tree", "a plant")
    assert first.resonance == 1.0
    again = hub.share_concept(b, "tree")
    assert again.resonance == 2.0
    assert set(again.contributors) == {"Alice", "Bob"}
    assert len(hub.concepts) == 1  # deduped by term


def test_concept_propagates_between_nodes():
    hub = CollectiveHub()
    alice = _node("Alice")
    bob = _node("Bob")
    hub.register_node(alice.name, node_id=alice.name)
    hub.register_node(bob.name, node_id=bob.name)

    alice.learning.teach("tree", "a living plant with a trunk", ["oak"])
    assert "tree" not in bob.learning.language.vocabulary

    # Alice shares; hub records; Bob absorbs.
    payload = alice.make_payload("tree")
    hub.share_concept(alice.name, payload["term"], payload["definition"], payload["examples"])
    newly = bob.absorb(payload)

    assert newly is True
    assert "tree" in bob.learning.language.vocabulary
    assert bob.learning.language.vocabulary["tree"]["source"] == "collective"
    assert bob.learning.language.vocabulary["tree"]["contributor"] == "Alice"


def test_node_does_not_reabsorb_its_own_concept():
    alice = _node("Alice")
    alice.learning.teach("tree", "a living plant")
    payload = alice.make_payload("tree")

    # Alice's own concept echoed back to her must be a no-op.
    assert alice.absorb(payload) is False


def test_emergence_requires_multiple_nodes():
    hub = CollectiveHub()
    solo = hub.register_node("Alice")
    hub.share_concept(solo, "tree", "a plant")
    hub.share_concept(solo, "river", "flowing water")

    # One node, however knowledgeable, is not a collective.
    assert hub.state()["emergence_level"] == 0
    assert hub.state()["collective_consciousness"] == 0.0

    hub.register_node("Bob")
    state = hub.state()
    assert state["node_count"] == 2
    assert state["emergence_level"] >= 1
    assert state["collective_consciousness"] > 0.0
