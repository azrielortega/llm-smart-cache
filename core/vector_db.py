import json
import os
import time

import faiss
import numpy as np


class VectorDB:
    """Owns a FAISS index plus the metadata and eviction bookkeeping (creation /
    last-access time) for each vector, as a single unit, so a vector, its
    metadata, and its eviction state can never drift out of sync.

    Eviction policy:
      - ttl_seconds: entries older than this (by creation time) are treated as
        expired on read, and are physically dropped on the next `add()`.
      - max_size: if set, `add()` trims down to the `max_size` most-recently-used
        entries (by insert time or last `touch()`) after every insert, so the
        index can't grow unbounded. `search()` alone does not count as a use.
    Both are optional and independent; leave either as None to disable it.
    """

    def __init__(self, dimension=384, index_path=None, metadata_path=None,
                 ttl_seconds=None, max_size=None):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self._records = self._load_records()

        if self.index.ntotal != len(self._records):
            raise RuntimeError(
                f"VectorDB is corrupted: index has {self.index.ntotal} vectors "
                f"but metadata has {len(self._records)} entries "
                f"({self.index_path!r}, {self.metadata_path!r})."
            )

    def add(self, vector, metadata):
        now = time.time()
        self.index.add(vector.astype('float32'))
        self._records.append({"metadata": metadata, "created_at": now, "last_accessed": now})
        self._evict()

    def search(self, vector, k=1):
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
        if not self.index_path or not self.metadata_path:
            raise ValueError("index_path/metadata_path not configured for VectorDB.save()")
        faiss.write_index(self.index, self.index_path)
        self._save_records()

    def _evict(self):
        """Drop TTL-expired and (if over max_size) least-recently-used entries,
        rebuilding the index from the survivors. IndexFlatL2 stores raw vectors,
        so `reconstruct` recovers them without needing to re-embed anything."""
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
            vectors = np.vstack([self.index.reconstruct(i) for i in keep])
            new_index.add(vectors)

        self.index = new_index
        self._records = [self._records[i] for i in keep]

    def _load_records(self):
        if not self.metadata_path or not os.path.exists(self.metadata_path):
            return []
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def _save_records(self):
        tmp_path = self.metadata_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._records, f)
        os.replace(tmp_path, self.metadata_path)
