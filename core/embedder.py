from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model {self.model_name!r}: {e}"
            ) from e

        self.dimension = self.model.get_embedding_dimension()
        if not self.dimension:
            raise RuntimeError(f"Embedding model {self.model_name!r} did not report its output dimension")

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Cannot embed empty or invalid text: {text!r}")

        try:
            return self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        except Exception as e:
            raise RuntimeError(f"Embedding failed for text {text!r}: {e}") from e
