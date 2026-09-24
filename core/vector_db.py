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

        self.index, self._records = self._load()

    def add(self, vector, metadata):
        """Store a vector with its metadata, then evict expired and over-capacity entries.

        Inputs:  vector (np.ndarray) - shape (1, dimension), metadata (dict) - returned by search() on a match
        """
        now = time.time()
        self.index.add(vector.astype('float32'))
        self._records.append({"metadata": metadata, "created_at": now, "last_accessed": now})
        self._evict()

    def search(self, vector, k=1):
        """Find the k nearest stored vectors, skipping TTL-expired ones. Does not count as a use for LRU.

        Inputs:  vector (np.ndarray) - shape (1, dimension), k (int) - neighbors to check
        Outputs: list[dict] - {"id", "distance", "metadata"}, nearest first; may be fewer than k
        """
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(vector.astype('float32'), k)

        now = time.time()
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = self._records[idx]
            if self.ttl_seconds is not None and (now - record["created_at"]) > self.ttl_seconds:
                continue  # expired: treat as a miss, physically dropped on next add()
            results.append({"id": int(idx), "distance": distance, "metadata": record["metadata"]})
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
            json.dump(self._records, f)
        os.replace(index_tmp, self.index_path)
        os.replace(metadata_tmp, self.metadata_path)

    def _evict(self):
        """Drop TTL-expired and (if over max_size) least recently used entries, rebuilding the index from the rest."""
        if self.index.ntotal == 0:
            return

        now = time.time()
        keep = list(range(self.index.ntotal))

        if self.ttl_seconds is not None:
            keep = [i for i in keep if (now - self._records[i]["created_at"]) <= self.ttl_seconds]

        if self.max_size is not None and len(keep) > self.max_size:
            keep.sort(key=lambda i: self._records[i]["last_accessed"], reverse=True)
            keep = keep[:self.max_size]
            keep.sort()  # restore original relative order for reconstruct

        if len(keep) == self.index.ntotal:
            return  # nothing to evict

        new_index = faiss.IndexFlatL2(self.dimension)
        if keep:
            # IndexFlatL2 stores raw vectors, so reconstruct() recovers them without re-embedding.
            vectors = np.vstack([self.index.reconstruct(i) for i in keep])
            new_index.add(vectors)

        self.index = new_index
        self._records = [self._records[i] for i in keep]

    def _load(self):
        """Load the index and metadata from disk, or start empty if they're missing or broken.

        Outputs: (faiss.Index, list) - the index and its matching records
        """
        index = faiss.IndexFlatL2(self.dimension)
        records = []
        try:
            if self.index_path and os.path.exists(self.index_path):
                index = faiss.read_index(self.index_path)
            if self.metadata_path and os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    records = json.load(f)
        except Exception as e:
            logger.warning("Could not read cache files (%s), starting with an empty cache.", e)
            return faiss.IndexFlatL2(self.dimension), []

        # Can happen if a crash landed between the two renames in save().
        if index.ntotal != len(records):
            logger.warning(
                "Cache files are out of sync (index has %d vectors, metadata has %d entries) "
                "in %r / %r, starting with an empty cache.",
                index.ntotal, len(records), self.index_path, self.metadata_path,
            )
            return faiss.IndexFlatL2(self.dimension), []

        return index, records
