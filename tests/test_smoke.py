"""
Smoke tests for the CHIMERA working core.

These are intentionally minimal: they prove the package imports cleanly and that
the OrganicLearningSystem can hold a conversation, learn a taught concept, and
serialize/restore its state. If these pass, `python run.py` has a working brain
behind it.
"""

import asyncio
import json

import pytest

from chimera_core.language.chimera_language_learning import OrganicLearningSystem


def test_package_imports_cleanly():
    # Importing the package root must not drag in broken optional subsystems.
    import chimera_core

    assert chimera_core.__version__


def test_interact_returns_expected_shape():
    chimera = OrganicLearningSystem("test")
    result = asyncio.run(chimera.interact("hello there"))

    for key in (
        "response",
        "understanding",
        "thoughts_formed",
        "words_known",
        "development_stage",
        "curiosity",
    ):
        assert key in result, f"missing key: {key}"

    assert isinstance(result["response"], str) and result["response"]
    assert result["thoughts_formed"] >= 1
    assert chimera.conversation_count == 1


def test_teaching_adds_vocabulary():
    chimera = OrganicLearningSystem("test")
    assert "tree" not in chimera.language.vocabulary

    result = chimera.teach(
        "tree",
        "A living plant with a trunk and leaves",
        ["Oak trees are tall", "Pine trees have needles"],
    )

    assert result["learned"] is True
    assert "tree" in chimera.language.vocabulary
    assert chimera.language.vocabulary["tree"]["taught"] is True


def test_state_roundtrip(tmp_path):
    chimera = OrganicLearningSystem("test")
    asyncio.run(chimera.interact("the sky is blue"))
    chimera.teach("sky", "The space above the earth")

    state_file = tmp_path / "state.json"
    chimera.save_state(str(state_file))
    assert state_file.exists()
    # File must be valid JSON.
    json.loads(state_file.read_text())

    revived = OrganicLearningSystem("placeholder")
    revived.load_state(str(state_file))

    assert revived.chimera_id == "test"
    assert "sky" in revived.language.vocabulary
    assert len(revived.reasoning.thought_network) == len(chimera.reasoning.thought_network)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
