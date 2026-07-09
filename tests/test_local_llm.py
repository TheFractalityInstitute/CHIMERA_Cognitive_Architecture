"""
Tests for the LocalLLM voice layer.

An injected transport stands in for Ollama, so these run without a local model.
The real end-to-end check is running Ollama on a device (docs/LOCAL_MODEL_SETUP.md).
"""

from chimera_core.language.local_llm import LocalLLM


def make_llm(handler):
    return LocalLLM(base_url="http://localhost:11434", model="gemma2:2b", http=handler)


def test_available_true_and_false():
    ok = make_llm(lambda url, payload, timeout: {"models": []})
    assert ok.available() is True

    def boom(url, payload, timeout):
        raise ConnectionError("refused")

    assert make_llm(boom).available() is False


def test_build_messages_grounds_in_state():
    llm = make_llm(lambda *a: {})
    messages = llm.build_messages(
        "Fred", "who are you?",
        words=["tree", "river"], feeling="I feel shaken up.",
        experiences=["learned 'dragon'"],
    )
    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "Fred" in system
    assert "tree" in system and "river" in system
    assert "shaken up" in system
    assert "dragon" in system
    assert messages[1] == {"role": "user", "content": "who are you?"}


def test_respond_parses_ollama_chat():
    def handler(url, payload, timeout):
        assert url.endswith("/api/chat")
        assert payload["model"] == "gemma2:2b"
        assert payload["stream"] is False
        return {"message": {"role": "assistant", "content": "  Hi! I'm still learning. "}}

    reply = make_llm(handler).respond("Fred", "hello")
    assert reply == "Hi! I'm still learning."


def test_respond_returns_none_on_error():
    def handler(url, payload, timeout):
        raise TimeoutError("slow")

    assert make_llm(handler).respond("Fred", "hello") is None


def test_respond_returns_none_on_empty_content():
    handler = lambda url, payload, timeout: {"message": {"content": "   "}}
    assert make_llm(handler).respond("Fred", "hello") is None


def test_words_are_capped_in_prompt():
    llm = make_llm(lambda *a: {})
    many = [f"word{i}" for i in range(100)]
    system = llm.build_messages("Fred", "hi", words=many)[0]["content"]
    assert "word0" in system
    assert "word99" not in system  # only the first 40 are included
