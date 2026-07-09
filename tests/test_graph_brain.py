"""
Tests for the GraphBrain Cypher/model logic.

These use an injected `runner` so they exercise query construction, parameter
binding, and result mapping without needing a live Neo4j database. (The real
end-to-end check is `python -m chimera_core.collective.graph_brain` against an
actual Aura instance.)
"""

from chimera_core.collective.graph_brain import GraphBrain, _slug


def make_brain(return_value=None):
    calls = []

    def runner(cypher, params):
        calls.append((cypher, params))
        return return_value if return_value is not None else []

    return GraphBrain(runner=runner, collective_name="The Collective"), calls


def test_available_with_runner():
    brain, _ = make_brain()
    assert brain.available() is True


def test_available_requires_credentials_without_runner():
    assert GraphBrain(uri=None, password=None).available() is False
    assert GraphBrain(uri="neo4j+s://x", password="pw").available() is True


def test_ensure_mind_links_to_collective():
    brain, calls = make_brain()
    mind_id = brain.ensure_mind("Fred")
    assert mind_id == "fred"
    cypher, params = calls[-1]
    assert params["id"] == "fred"
    assert params["name"] == "Fred"
    assert params["coll"] == "the-collective"
    assert "PART_OF" in cypher


def test_remember_concept_writes_mind_and_collective():
    brain, calls = make_brain()
    brain.remember_concept("fred", "Dragon", "a big creature", felt="shaken", teacher="Grazi")
    cypher, params = calls[-1]
    # Term is normalized; both the mind and the collective are targets.
    assert params["term"] == "dragon"
    assert params["mind"] == "fred"
    assert params["coll"] == "the-collective"
    assert params["teacher"] == "Grazi"
    assert params["felt"] == "shaken"
    assert "(m)-[:EXPERIENCED]->(e)" in cypher
    assert "(coll)-[:EXPERIENCED]->(e)" in cypher
    assert "(e)-[:ABOUT]->(c)" in cypher


def test_record_episode_params():
    brain, calls = make_brain()
    brain.record_episode("fred", kind="sensed", text="picked up", felt="calm")
    _, params = calls[-1]
    assert params["kind"] == "sensed"
    assert params["text"] == "picked up"
    assert params["mind"] == "fred"
    assert "eid" in params and "t" in params


def test_known_concepts_maps_rows():
    rows = [{"term": "dragon", "definition": "a big creature", "confidence": 0.7, "source": "taught"}]
    brain, _ = make_brain(return_value=rows)
    out = brain.known_concepts("fred")
    assert out == rows


def test_collective_state_defaults_when_empty():
    brain, _ = make_brain(return_value=[])
    state = brain.collective_state()
    assert state == {"minds": 0, "concepts": 0, "episodes": 0}


def test_slug():
    assert _slug("Dante's CHIMERA") == "dante-s-chimera"
    assert _slug("") == "mind"
