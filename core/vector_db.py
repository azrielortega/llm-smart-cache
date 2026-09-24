import heapq
import json
import logging
import os
import time

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorDB:
    """FAISS index plus per-vector metadata and timestamps, kept as one unit so they never drift apart.
    Optional eviction: `ttl_seconds` (age since insert) and `max_size` (least recently used); None disables either."""

    def __init__(self, dimension=384, index_path=None, metadata_path=None,
                 ttl_seconds=None, max_size=None):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        # _records maps each stable FAISS id to its entry, so eviction can remove
        # entries by id instead of rebuilding the index around them.
        self.index, self._records = self._load()
        self._next_id = max(self._records, default=-1) + 1

    def add(self, vector, metadata):
        """Store a vector with its metadata, then evict expired and over-capacity entries.

        Inputs:  vector (np.ndarray) - shape (1, dimension), metadata (dict) - returned by search() on a match
        """
        now = time.time()
        record_id = self._next_id
        self._next_id += 1
        self.index.add_with_ids(vector.astype('float32'), np.array([record_id], dtype='int64'))
        self._records[record_id] = {"metadata": metadata, "created_at": now, "last_accessed": now}
        self._evict()

    def search(self, vector, k=1):
        """Find the k nearest stored vectors, skipping TTL-expired ones. Does not count as a use for LRU.

        Inputs:  vector (np.ndarray) - shape (1, dimension), k (int) - neighbors to check
        Outputs: list[dict] - {"id", "distance", "metadata"}, nearest first; may be fewer than k
        """
        if self.index.ntotal == 0:
            return []

        distances, ids = self.index.search(vector.astype('float32'), k)

        now = time.time()
        results = []
        for distance, record_id in zip(distances[0], ids[0]):
            if record_id == -1:
                continue
            record = self._records[int(record_id)]
            if self.ttl_seconds is not None and (now - record["created_at"]) > self.ttl_seconds:
                continue  # expired: treat as a miss, physically dropped on next add()
            results.append({"id": int(record_id), "distance": distance, "metadata": record["metadata"]})
        return results

    def touch(self, record_id):
        """Mark an entry as used, so LRU eviction keeps it.

        Inputs:  record_id (int) - the "id" from a search() result
        """
        self._records[record_id]["last_accessed"] = time.time()

    def save(self):
        """Write the index and metadata to disk atomically; raises ValueError if paths aren't configured."""
        if not self.index_path or not self.metadata_path:
            raise ValueError("index_path/metadata_path not configured for VectorDB.save()")
        # Write both to temp files first, then swap them in, so a crash mid-write
        # never leaves a half-written file behind.
        index_tmp = self.index_path + ".tmp"
        metadata_tmp = self.metadata_path + ".tmp"
        faiss.write_index(self.index, index_tmp)
        with open(metadata_tmp, "w") as f:
            json.dump(self._records, f)  # int ids become string keys; _load() converts them back
        os.replace(index_tmp, self.index_path)
        os.replace(metadata_tmp, self.metadata_path)

    def _evict(self):
        """Remove TTL-expired and (if over max_size) least recently used entries from the index by id."""
        now = time.time()
        drop = []
        if self.ttl_seconds is not None:
            drop = [i for i, r in self._records.items() if (now - r["created_at"]) > self.ttl_seconds]

        if self.max_size is not None and len(self._records) - len(drop) > self.max_size:
            excess = len(self._records) - len(drop) - self.max_size
            dropped = set(drop)
            survivors = (i for i in self._records if i not in dropped)
            drop += heapq.nsmallest(excess, survivors, key=lambda i: self._records[i]["last_accessed"])

        if not drop:
            return
        self.index.remove_ids(np.array(drop, dtype='int64'))
        for record_id in drop:
            del self._records[record_id]

    def _empty_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def _load(self):
        """Load the index and metadata from disk, or start empty if they're missing, broken or an older format.

        Outputs: (faiss.IndexIDMap, dict) - the index and its records keyed by FAISS id
        """
        index = self._empty_index()
        records = {}
        try:
            if self.index_path and os.path.exists(self.index_path):
                index = faiss.read_index(self.index_path)
            if self.metadata_path and os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    records = json.load(f)
        except Exception as e:
            logger.warning("Could not read cache files (%s), starting with an empty cache.", e)
            return self._empty_index(), {}

        # Caches saved before entries had stable ids: a plain IndexFlatL2 plus a JSON list.
        if not isinstance(index, faiss.IndexIDMap) or not isinstance(records, dict):
            if index.ntotal or records:
                logger.warning("Cache in %r / %r uses an older format, starting with an empty cache.",
                               self.index_path, self.metadata_path)
            return self._empty_index(), {}

        records = {int(record_id): record for record_id, record in records.items()}

        # Can happen if a crash landed between the two renames in save().
        if set(faiss.vector_to_array(index.id_map).tolist()) != set(records):
            logger.warning(
                "Cache files are out of sync (index has %d vectors, metadata has %d entries) "
                "in %r / %r, starting with an empty cache.",
                index.ntotal, len(records), self.index_path, self.metadata_path,
            )
            return self._empty_index(), {}

        return index, records
