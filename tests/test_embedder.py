import numpy as np
import pytest

from core.embedder import Embedder


def test_encode_returns_expected_shape(fake_sentence_transformer):
    embedder = Embedder(model_name="fake-model")
    vector = embedder.encode("hello world")
    assert vector.shape == (1, 384)


def test_encode_is_deterministic_for_same_text(fake_sentence_transformer):
    embedder = Embedder(model_name="fake-model")
    assert np.array_equal(embedder.encode("hello"), embedder.encode("hello"))


def test_encode_differs_for_different_text(fake_sentence_transformer):
    embedder = Embedder(model_name="fake-model")
    assert not np.array_equal(embedder.encode("hello"), embedder.encode("goodbye"))


@pytest.mark.parametrize("bad_input", ["", "   ", None, 123, [], {}])
def test_encode_rejects_invalid_text(fake_sentence_transformer, bad_input):
    embedder = Embedder(model_name="fake-model")
    with pytest.raises(ValueError):
        embedder.encode(bad_input)


def test_init_wraps_model_load_failure(monkeypatch):
    def broken_loader(model_name):
        raise OSError("model not found")

    monkeypatch.setattr("core.embedder.SentenceTransformer", broken_loader)
    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        Embedder(model_name="nonexistent-model")


def test_encode_wraps_model_encode_failure(monkeypatch):
    class BrokenModel:
        def __init__(self, model_name):
            pass

        def get_embedding_dimension(self):
            return 384 # Dummy Dimension Value

        def encode(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr("core.embedder.SentenceTransformer", BrokenModel)
    embedder = Embedder(model_name="fake-model")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        embedder.encode("hello")


def test_defaults_to_configured_model_name(fake_sentence_transformer, monkeypatch):
    monkeypatch.setattr("core.embedder.EMBEDDING_MODEL_NAME", "configured-default")
    embedder = Embedder()
    assert embedder.model_name == "configured-default"


def test_explicit_model_name_overrides_config(fake_sentence_transformer, monkeypatch):
    monkeypatch.setattr("core.embedder.EMBEDDING_MODEL_NAME", "configured-default")
    embedder = Embedder(model_name="explicit-model")
    assert embedder.model_name == "explicit-model"


def test_dimension_is_read_from_model(fake_sentence_transformer, monkeypatch):
    monkeypatch.setattr(
        "core.embedder.SentenceTransformer",
        lambda name: fake_sentence_transformer(name, dimension=16),
    )
    assert Embedder(model_name="small-model").dimension == 16
