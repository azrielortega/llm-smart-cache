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
    happens when the nearest stored question's distance to the incoming query is
    below `max_distance`. Lower `max_distance` = stricter matching (fewer hits,
    less risk of returning a wrong cached answer for a different question);
    higher = looser matching (more hits, more risk of false positives).
    """

    def __init__(self, max_distance=None, cache_dir="cache_data",
                 ttl_seconds=None, max_size=1000, model_name=None,
                 embedding_dimension=None):
        self.embedder = Embedder(model_name=model_name)
        self.max_distance = CACHE_MAX_DISTANCE if max_distance is None else max_distance
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.db = VectorDB(
            dimension=embedding_dimension or EMBEDDING_DIMENSION,
            index_path=os.path.join(cache_dir, "index.faiss"),
            metadata_path=os.path.join(cache_dir, "metadata.json"),
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )

    def query(self, user_text):
        query_vector = self.embedder.encode(user_text)
        results = self.db.search(query_vector)

        if results and results[0]["distance"] < self.max_distance:
            logger.debug("Cache HIT (distance=%.4f) for: %r", results[0]["distance"], user_text)
            return results[0]["metadata"]["answer"]

        logger.debug("Cache MISS for: %r", user_text)
        return None

    def update(self, question, answer):
        vector = self.embedder.encode(question)
        self.db.add(vector, {"question": question, "answer": answer})
        self.db.save()

    def save(self):
        self.db.save()
