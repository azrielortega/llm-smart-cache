import pytest

from core.cache_logic import SmartCache
from core.llm_client import get_or_call
from core.mock_llm import MockClient


@pytest.fixture
def cache(tmp_path, fake_sentence_transformer):
    return SmartCache(
        cache_dir=str(tmp_path / "cache"),
        max_distance=0.05,
        embedding_dimension=384,
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
