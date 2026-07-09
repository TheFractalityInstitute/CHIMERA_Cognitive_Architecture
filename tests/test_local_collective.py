"""
Tests for LocalCollective — the in-process coordinator behind the web UI.

Proves the family experience: teach a word on one person's node and it flows to
the others, with attribution, and persists to disk.
"""

from chimera_core.collective.local import LocalCollective


def test_teach_and_share_propagates(tmp_path):
    col = LocalCollective(persist_dir=tmp_path)
    col.get_or_create("Dante's CHIMERA")
    col.get_or_create("Dad's CHIMERA")

    result, events = col.teach_and_share(
        "Dante's CHIMERA", "dragon", "a big fire-breathing creature", ["Smaug is a dragon"]
    )

    assert result["learned"] is True
    assert len(events) == 1
    ev = events[0]
    assert ev["to"] == "Dad's CHIMERA"
    assert ev["from"] == "Dante's CHIMERA"
    assert ev["term"] == "dragon"

    dad = col.nodes["Dad's CHIMERA"].learning.language.vocabulary
    assert "dragon" in dad
    assert dad["dragon"]["source"] == "collective"
    assert dad["dragon"]["contributor"] == "Dante's CHIMERA"


def test_no_event_for_solo_node(tmp_path):
    col = LocalCollective(persist_dir=tmp_path)
    col.get_or_create("Solo")
    _, events = col.teach_and_share("Solo", "tree", "a plant")
    assert events == []  # nobody else to share with


def test_state_and_concepts(tmp_path):
    col = LocalCollective(persist_dir=tmp_path)
    col.get_or_create("A")
    col.get_or_create("B")
    col.teach_and_share("A", "river", "flowing water")

    state = col.state()
    assert state["node_count"] == 2
    assert state["concept_count"] == 1
    assert state["emergence_level"] >= 1

    b_concepts = {c["term"]: c for c in col.concepts_for("B")}
    assert "river" in b_concepts
    assert b_concepts["river"]["source"] == "collective"


def test_persistence_across_instances(tmp_path):
    col1 = LocalCollective(persist_dir=tmp_path)
    col1.teach("Dante's CHIMERA", "castle", "a big stone fortress")

    # A fresh collective pointed at the same dir should reload the node's words.
    col2 = LocalCollective(persist_dir=tmp_path)
    node = col2.get_or_create("Dante's CHIMERA")
    assert "castle" in node.learning.language.vocabulary
