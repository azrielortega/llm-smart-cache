import hashlib
import os
import sys
import types

import numpy as np
import pytest

# Make sure `core` is importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sentence-transformers (and its torch dependency) isn't needed to test our
# logic - only its interface is. If it's not installed, stub it out so
# `core.embedder` can still be imported; tests below replace
# `core.embedder.SentenceTransformer` with a deterministic fake either way.
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = None
        sys.modules["sentence_transformers"] = stub

from core.embedder import Embedder  # noqa: E402  (must come after the stub above)


class FakeSentenceTransformer:
    """Deterministic, dependency-free stand-in for
    sentence_transformers.SentenceTransformer.

    Hashes each text to seed a fixed-size random vector, so identical text
    always encodes to the identical vector and different text encodes to a
    (very likely) different one - without downloading a real model.
    """

    def __init__(self, model_name, dimension=384):
        self.model_name = model_name
        self.dimension = dimension

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vectors.append(rng.rand(self.dimension).astype("float32"))
        return np.array(vectors)


class FakeEmbedder(Embedder):
    """Real Embedder (same encode() validation) around FakeSentenceTransformer,
    built without loading a model, so it can be passed to SmartCache(embedder=...)."""

    def __init__(self, model_name="fake-model", dimension=384):
        self.model_name = model_name
        self.model = FakeSentenceTransformer(model_name, dimension)
        self.dimension = dimension


@pytest.fixture
def fake_embedder():
    """The FakeEmbedder class, e.g. SmartCache(embedder=fake_embedder("model-a"))."""
    return FakeEmbedder


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    """Patch core.embedder.SentenceTransformer with the fake above."""
    monkeypatch.setattr("core.embedder.SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer
