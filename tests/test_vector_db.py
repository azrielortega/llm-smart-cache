import json
import time

import numpy as np
import pytest

from core.vector_db import VectorDB

DIM = 8


def make_vector(seed):
    return np.random.RandomState(seed).rand(1, DIM).astype("float32")


def test_add_and_search_returns_nearest_match():
    db = VectorDB(dimension=DIM)
    v1, v2 = make_vector(1), make_vector(2)
    db.add(v1, {"answer": "first"})
    db.add(v2, {"answer": "second"})

    results = db.search(v1)
    assert results[0]["metadata"]["answer"] == "first"
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-4)


def test_search_on_empty_index_returns_empty_list():
    db = VectorDB(dimension=DIM)
    assert db.search(make_vector(1)) == []


def test_index_and_metadata_stay_in_sync_after_add():
    db = VectorDB(dimension=DIM)
    db.add(make_vector(1), {"answer": "a"})
    db.add(make_vector(2), {"answer": "b"})
    assert db.index.ntotal == len(db._records) == 2


def test_save_and_load_round_trip(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")

    db = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    db.add(make_vector(1), {"answer": "first"})
    db.save()

    reloaded = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    assert reloaded.index.ntotal == 1
    results = reloaded.search(make_vector(1))
    assert results[0]["metadata"]["answer"] == "first"


def test_save_without_paths_raises():
    db = VectorDB(dimension=DIM)
    with pytest.raises(ValueError):
        db.save()


def test_index_metadata_mismatch_starts_empty(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")

    db = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    db.add(make_vector(1), {"answer": "first"})
    db.save()

    # Corrupt: metadata now claims 2 entries but the saved index only has 1 vector.
    with open(metadata_path, "w") as f:
        json.dump([{"metadata": {}, "created_at": 0, "last_accessed": 0}] * 2, f)

    db = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    assert db.index.ntotal == 0
    assert db.search(make_vector(1)) == []


def test_unreadable_metadata_starts_empty(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.json")

    db = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    db.add(make_vector(1), {"answer": "first"})
    db.save()

    with open(metadata_path, "w") as f:
        f.write('[{"metadata": ')  # truncated JSON, as if a write was cut off

    db = VectorDB(dimension=DIM, index_path=index_path, metadata_path=metadata_path)
    assert db.index.ntotal == 0


def test_save_leaves_no_temp_files(tmp_path):
    db = VectorDB(dimension=DIM, index_path=str(tmp_path / "index.faiss"),
                  metadata_path=str(tmp_path / "metadata.json"))
    db.add(make_vector(1), {"answer": "first"})
    db.save()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.faiss", "metadata.json"]


def test_ttl_expired_entry_is_excluded_from_search():
    db = VectorDB(dimension=DIM, ttl_seconds=100)
    v = make_vector(1)
    db.add(v, {"answer": "stale"})
    db._records[0]["created_at"] = time.time() - 1000  # force expiry

    assert db.search(v) == []


def test_ttl_expired_entry_is_dropped_on_next_add():
    db = VectorDB(dimension=DIM, ttl_seconds=100)
    db.add(make_vector(1), {"answer": "stale"})
    db._records[0]["created_at"] = time.time() - 1000  # force expiry

    db.add(make_vector(2), {"answer": "fresh"})

    assert db.index.ntotal == 1
    assert db._records[0]["metadata"]["answer"] == "fresh"


def test_max_size_evicts_least_recently_used():
    db = VectorDB(dimension=DIM, max_size=2)
    v1, v2, v3 = make_vector(1), make_vector(2), make_vector(3)
    db.add(v1, {"answer": "one"})
    db.add(v2, {"answer": "two"})

    # Make v1 look more recently used than v2, without depending on real-time
    # ordering between add() calls.
    db._records[0]["last_accessed"] = time.time() + 100
    db._records[1]["last_accessed"] = time.time() - 100

    db.add(v3, {"answer": "three"})

    assert db.index.ntotal == 2
    remaining = {record["metadata"]["answer"] for record in db._records}
    assert remaining == {"one", "three"}
