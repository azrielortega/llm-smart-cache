import time

import pytest

from core.cache_logic import SmartCache


@pytest.fixture
def cache(tmp_path, fake_embedder):
    return SmartCache(
        embedder=fake_embedder(),
        cache_dir=str(tmp_path / "cache"),
        max_distance=0.05,
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


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_update_skips_empty_answer(cache, answer):
    cache.update("How to bake a cake?", answer)
    assert cache.db.index.ntotal == 0


def test_update_persists_to_disk(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    original = SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, max_distance=0.05)
    original.update("How to bake a cake?", "Preheat the oven to 350F.")

    reloaded = SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, max_distance=0.05)
    assert reloaded.query("How to bake a cake?") == "Preheat the oven to 350F."


def test_ttl_expired_nearest_match_falls_through_to_next_fresh_match(tmp_path, fake_embedder):
    cache = SmartCache(
        embedder=fake_embedder(),
        cache_dir=str(tmp_path / "cache"),
        max_distance=0.05,
        ttl_seconds=100,
    )
    query_vector = cache.embedder.encode("How to bake a cake?")
    near_vector = query_vector.copy()
    near_vector[0][0] += 0.01  # tiny perturbation: still well within max_distance

    # Nearest match (distance 0) - forced expired *after* both adds so it's
    # still physically in the index (eviction is deferred to the next add()).
    cache.db.add(query_vector, {"question": "exact", "answer": "stale answer"})
    # Second-nearest match: close but not exact, and stays fresh.
    cache.db.add(near_vector, {"question": "near", "answer": "fresh answer"})
    cache.db._records[0]["created_at"] = time.time() - 10_000

    assert cache.query("How to bake a cake?") == "fresh answer"


def test_only_the_hit_entry_is_marked_as_used(tmp_path, fake_embedder):
    cache = SmartCache(embedder=fake_embedder(), cache_dir=str(tmp_path / "cache"), max_distance=0.05)
    cache.update("How to bake a cake?", "Preheat the oven to 350F.")
    cache.update("What is the capital of France?", "Paris.")
    for record in cache.db._records.values():
        record["last_accessed"] = 0

    assert cache.query("How to bake a cake?") == "Preheat the oven to 350F."

    hit, other = cache.db._records.values()
    assert hit["last_accessed"] > 0
    assert other["last_accessed"] == 0  # returned by search() as a neighbor, but not a hit


def test_miss_does_not_mark_anything_as_used(cache):
    cache.update("How to bake a cake?", "Preheat the oven to 350F.")
    cache.db._records[0]["last_accessed"] = 0

    assert cache.query("What is the capital of France?") is None
    assert cache.db._records[0]["last_accessed"] == 0


def test_reload_with_same_embedding_model_works(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    SmartCache(embedder=fake_embedder("model-a"), cache_dir=cache_dir).update("How to bake a cake?", "Preheat.")

    reloaded = SmartCache(embedder=fake_embedder("model-a"), cache_dir=cache_dir, max_distance=0.05)
    assert reloaded.query("How to bake a cake?") == "Preheat."


def test_reload_with_different_embedding_model_is_refused(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    SmartCache(embedder=fake_embedder("model-a"), cache_dir=cache_dir).update("How to bake a cake?", "Preheat.")

    with pytest.raises(RuntimeError, match="model-a"):
        SmartCache(embedder=fake_embedder("model-b"), cache_dir=cache_dir)


def test_empty_cache_can_switch_embedding_model(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    SmartCache(embedder=fake_embedder("model-a"), cache_dir=cache_dir)

    SmartCache(embedder=fake_embedder("model-b"), cache_dir=cache_dir)
    with open(tmp_path / "cache" / "embedding_model.txt") as f:
        assert f.read() == "model-b"


def test_reload_with_different_llm_model_is_refused(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, llm_model_name="llm-a").update("How to bake a cake?", "Preheat.")

    with pytest.raises(RuntimeError, match="LLM model 'llm-a'"):
        SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, llm_model_name="llm-b")


def test_reload_with_same_llm_model_works(tmp_path, fake_embedder):
    cache_dir = str(tmp_path / "cache")
    SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, llm_model_name="llm-a").update("How to bake a cake?", "Preheat.")

    reloaded = SmartCache(embedder=fake_embedder(), cache_dir=cache_dir, llm_model_name="llm-a", max_distance=0.05)
    assert reloaded.query("How to bake a cake?") == "Preheat."


def test_defaults_come_from_config(tmp_path, fake_embedder, monkeypatch):
    monkeypatch.setattr("core.cache_logic.CACHE_MAX_DISTANCE", 0.42)
    monkeypatch.setattr("core.cache_logic.LLM_MODEL_NAME", "llm-x")

    cache = SmartCache(embedder=fake_embedder(), cache_dir=str(tmp_path / "cache"))

    assert cache.max_distance == 0.42
    assert cache.llm_model_name == "llm-x"


def test_index_dimension_comes_from_embedding_model(tmp_path, fake_embedder):
    cache = SmartCache(embedder=fake_embedder(dimension=16), cache_dir=str(tmp_path / "cache"), max_distance=0.05)
    cache.update("How to bake a cake?", "Preheat.")

    assert cache.db.dimension == 16
    assert cache.query("How to bake a cake?") == "Preheat."


def test_builds_its_own_embedder_when_none_is_passed(tmp_path, fake_sentence_transformer):
    cache = SmartCache(cache_dir=str(tmp_path / "cache"), model_name="model-a")
    assert cache.embedder.model_name == "model-a"
