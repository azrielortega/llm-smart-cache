import pytest

from core.cache_logic import SmartCache


@pytest.fixture
def cache(tmp_path, fake_sentence_transformer):
    return SmartCache(
        cache_dir=str(tmp_path / "cache"),
        max_distance=0.05,
        embedding_dimension=384,
    )


def test_miss_on_empty_cache(cache):
    assert cache.query("How do I bake a cake?") is None


def test_hit_on_identical_question(cache):
    cache.update("How to bake a cake?", "Preheat the oven to 350F.")
    assert cache.query("How to bake a cake?") == "Preheat the oven to 350F."


def test_miss_when_no_similar_question_cached(cache):
    cache.update("How to bake a cake?", "Preheat the oven to 350F.")
    assert cache.query("What is the capital of France?") is None


def test_query_rejects_empty_text(cache):
    with pytest.raises(ValueError):
        cache.query("")


def test_update_rejects_empty_question(cache):
    with pytest.raises(ValueError):
        cache.update("   ", "some answer")


def test_update_persists_to_disk(tmp_path, fake_sentence_transformer):
    cache_dir = str(tmp_path / "cache")
    original = SmartCache(cache_dir=cache_dir, max_distance=0.05, embedding_dimension=384)
    original.update("How to bake a cake?", "Preheat the oven to 350F.")

    reloaded = SmartCache(cache_dir=cache_dir, max_distance=0.05, embedding_dimension=384)
    assert reloaded.query("How to bake a cake?") == "Preheat the oven to 350F."


def test_defaults_come_from_config(tmp_path, fake_sentence_transformer, monkeypatch):
    monkeypatch.setattr("core.cache_logic.CACHE_MAX_DISTANCE", 0.42)
    monkeypatch.setattr("core.cache_logic.EMBEDDING_DIMENSION", 16)

    cache = SmartCache(cache_dir=str(tmp_path / "cache"))

    assert cache.max_distance == 0.42
    assert cache.db.dimension == 16
