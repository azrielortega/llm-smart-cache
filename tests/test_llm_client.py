from types import SimpleNamespace

import pytest

from core.cache_logic import SmartCache
from core.llm_client import OPENROUTER_BASE_URL, build_client, call_llm, get_or_call
from core.mock_llm import MockClient


@pytest.fixture
def cache(tmp_path, fake_embedder):
    return SmartCache(
        embedder=fake_embedder(),
        cache_dir=str(tmp_path / "cache"),
        max_distance=0.05,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("core.mock_llm.time.sleep", lambda _: None)
    return MockClient()


def test_miss_calls_llm_and_caches_answer(cache, client):
    answer, completion = get_or_call(cache, client, "How to bake a cake?")

    assert completion is not None
    assert answer == completion.choices[0].message.content
    assert cache.query("How to bake a cake?") == answer


def test_hit_returns_cached_answer_without_llm_call(cache, client):
    cache.update("How to bake a cake?", "Preheat the oven to 350F.")

    answer, completion = get_or_call(cache, client, "How to bake a cake?")

    assert completion is None
    assert answer == "Preheat the oven to 350F."


def test_none_answer_is_not_cached(cache, client, monkeypatch):
    completion = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    completion.choices[0].message.content = None
    monkeypatch.setattr("core.llm_client.call_llm", lambda *_: completion)

    get_or_call(cache, client, "How to bake a cake?")
    answer, completion = get_or_call(cache, client, "How to bake a cake?")

    assert answer is None
    assert completion is not None
    assert cache.db.index.ntotal == 0


def fake_client(create):
    """Minimal OpenAI-shaped client whose chat.completions.create is the given function."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_build_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_KEY"):
        build_client()


def test_build_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")
    client = build_client()
    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert client.api_key == "test-key"


def test_call_llm_defaults_to_configured_model(monkeypatch):
    monkeypatch.setattr("core.llm_client.LLM_MODEL_NAME", "configured-model")
    calls = []
    client = fake_client(lambda **kwargs: calls.append(kwargs) or "completion")

    assert call_llm(client, "Hello?") == "completion"
    assert calls == [{"model": "configured-model", "messages": [{"role": "user", "content": "Hello?"}]}]


def test_call_llm_wraps_client_errors():
    def broken_create(**kwargs):
        raise ConnectionError("network down")

    with pytest.raises(RuntimeError, match="LLM call failed for model 'some-model'"):
        call_llm(fake_client(broken_create), "Hello?", model="some-model")
