import logging
import os

from core.config import CACHE_MAX_DISTANCE, LLM_MODEL_NAME
from core.embedder import Embedder
from core.vector_db import VectorDB

logger = logging.getLogger(__name__)


class SmartCache:
    """Semantic cache in front of an LLM call: a question hits when a stored one is
    within `max_distance` (squared L2, lower = stricter) among its `search_k` nearest fresh neighbors."""

    def __init__(self, max_distance=None, cache_dir="cache_data",
                 ttl_seconds=None, max_size=1000, model_name=None,
                 search_k=5, llm_model_name=None, embedder=None, save_every=10):
        # A passed-in embedder (e.g. a test fake) takes precedence over model_name.
        self.embedder = embedder or Embedder(model_name=model_name)
        self.llm_model_name = llm_model_name or LLM_MODEL_NAME
        self.max_distance = CACHE_MAX_DISTANCE if max_distance is None else max_distance
        self.cache_dir = cache_dir
        self.search_k = search_k
        # Saving rewrites the whole index + metadata, so batch it: every `save_every`
        # writes (None = only on explicit save()). Callers save() on shutdown.
        self.save_every = save_every
        self._unsaved_writes = 0
        os.makedirs(cache_dir, exist_ok=True)

        self.db = VectorDB(
            dimension=self.embedder.dimension,
            index_path=os.path.join(cache_dir, "index.faiss"),
            metadata_path=os.path.join(cache_dir, "metadata.json"),
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )
        # Vectors from different embedding models aren't comparable, even at the
        # same dimension; answers from a different LLM would be silently stale.
        self._check_model("embedding_model.txt", "embedding", self.embedder.model_name)
        self._check_model("llm_model.txt", "LLM", self.llm_model_name)

    def _check_model(self, filename, kind, current):
        """Refuse to reuse a non-empty cache built with a different model, then record the current one.

        Inputs:  filename (str) - record file in cache_dir, kind (str) - label for the error, current (str) - model in use
        """
        path = os.path.join(self.cache_dir, filename)

        if os.path.exists(path) and self.db.index.ntotal > 0:
            with open(path) as f:
                saved = f.read().strip()
            if saved != current:
                raise RuntimeError(
                    f"Cache in {self.cache_dir!r} was built with {kind} model {saved!r}, "
                    f"but the current model is {current!r}. Delete that folder or use a "
                    f"different cache_dir."
                )

        with open(path, "w") as f:
            f.write(current)

    def query(self, user_text):
        """Look up the cached answer for the closest similar question.

        Inputs:  user_text (str) - the incoming question; raises ValueError if empty
        Outputs: str | None - the cached answer on a hit, None on a miss
        """
        query_vector = self.embedder.encode(user_text)
        results = self.db.search(query_vector, k=self.search_k)

        for result in results:
            if result["distance"] < self.max_distance:
                self.db.touch(result["id"])
                logger.debug("Cache HIT (distance=%.4f) for: %r", result["distance"], user_text)
                return result["metadata"]["answer"]

        logger.debug("Cache MISS for: %r", user_text)
        return None

    def update(self, question, answer):
        """Cache an answer for a question, saving to disk every `save_every` writes; skips empty answers.

        Inputs:  question (str), answer (str | None) - None/blank (e.g. refusals, filtered output) is not stored
        """
        # A stored None/blank answer would read back as a miss on every lookup,
        # adding a duplicate entry each time instead of ever being a hit.
        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Not caching empty answer for: %r", question)
            return

        vector = self.embedder.encode(question)
        self.db.add(vector, {"question": question, "answer": answer})
        self._unsaved_writes += 1
        if self.save_every and self._unsaved_writes >= self.save_every:
            self.save()

    def save(self):
        """Write the cache to disk. Call before exiting, or up to `save_every - 1` answers are lost."""
        self.db.save()
        self._unsaved_writes = 0
