import logging
import os

from core.config import CACHE_MAX_DISTANCE, EMBEDDING_DIMENSION
from core.embedder import Embedder
from core.vector_db import VectorDB

logger = logging.getLogger(__name__)


class SmartCache:
    """Semantic cache in front of an LLM call.

    Backed by FAISS IndexFlatL2, which returns *squared* L2 distance between
    embeddings - smaller distance means more semantically similar. A cache hit
    happens when the closest of the top `search_k` stored questions (skipping
    any that are TTL-expired) has a distance to the incoming query below
    `max_distance`. Lower `max_distance` = stricter matching (fewer hits, less
    risk of returning a wrong cached answer for a different question); higher =
    looser matching (more hits, more risk of false positives). `search_k` only
    matters when `ttl_seconds` is set - without TTL, the single nearest
    neighbor is always the best candidate.
    """

    def __init__(self, max_distance=None, cache_dir="cache_data",
                 ttl_seconds=None, max_size=1000, model_name=None,
                 embedding_dimension=None, search_k=5):
        self.embedder = Embedder(model_name=model_name)
        self.max_distance = CACHE_MAX_DISTANCE if max_distance is None else max_distance
        self.cache_dir = cache_dir
        self.search_k = search_k
        os.makedirs(cache_dir, exist_ok=True)

        self.db = VectorDB(
            dimension=embedding_dimension or EMBEDDING_DIMENSION,
            index_path=os.path.join(cache_dir, "index.faiss"),
            metadata_path=os.path.join(cache_dir, "metadata.json"),
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )
        self._check_embedding_model()

    def _check_embedding_model(self):
        """Refuse to reuse a cache built by a different embedding model, then record the current one.

        Vectors from different models aren't comparable, even when they have the
        same dimension, so mixing them would silently produce wrong hits.
        """
        path = os.path.join(self.cache_dir, "embedding_model.txt")
        current = self.embedder.model_name

        if os.path.exists(path) and self.db.index.ntotal > 0:
            with open(path) as f:
                saved = f.read().strip()
            if saved != current:
                raise RuntimeError(
                    f"Cache in {self.cache_dir!r} was built with embedding model {saved!r}, "
                    f"but the current model is {current!r}. Delete that folder or use a "
                    f"different cache_dir."
                )

        with open(path, "w") as f:
            f.write(current)

    def query(self, user_text):
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
        """Cache an answer for a question, skipping empty answers.

        Inputs:  question (str), answer (str | None) - None/blank (e.g. refusals, filtered output) is not stored
        """
        # A stored None/blank answer would read back as a miss on every lookup,
        # adding a duplicate entry each time instead of ever being a hit.
        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Not caching empty answer for: %r", question)
            return

        vector = self.embedder.encode(question)
        self.db.add(vector, {"question": question, "answer": answer})
        self.db.save()

    def save(self):
        self.db.save()
