import os

from core.embedder import Embedder
from core.vector_db import VectorDB


class SmartCache:
    """Semantic cache in front of an LLM call.

    Backed by FAISS IndexFlatL2, which returns *squared* L2 distance between
    embeddings - smaller distance means more semantically similar. A cache hit
    happens when the nearest stored question's distance to the incoming query is
    below `max_distance`. Lower `max_distance` = stricter matching (fewer hits,
    less risk of returning a wrong cached answer for a different question);
    higher = looser matching (more hits, more risk of false positives).
    """

    def __init__(self, max_distance=0.3, cache_dir="cache_data",
                 ttl_seconds=None, max_size=1000):
        self.embedder = Embedder()
        self.max_distance = max_distance
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.db = VectorDB(
            index_path=os.path.join(cache_dir, "index.faiss"),
            metadata_path=os.path.join(cache_dir, "metadata.json"),
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )

    def query(self, user_text):
        query_vector = self.embedder.encode(user_text)
        results = self.db.search(query_vector)

        if results and results[0]["distance"] < self.max_distance:
            print("Cache HIT!")
            return results[0]["metadata"]["answer"]

        print("Cache MISS...")
        return None

    def update(self, question, answer):
        vector = self.embedder.encode(question)
        self.db.add(vector, {"question": question, "answer": answer})
        self.db.save()

    def save(self):
        self.db.save()
