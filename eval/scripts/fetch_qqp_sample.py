"""One-off: samples labeled question pairs from the Quora Question Pairs
dataset (via Hugging Face's datasets-server API) and writes them to
eval/data/qqp_pairs.json for tune_threshold.py to consume.

Usage:
    python eval/scripts/fetch_qqp_sample.py
    python eval/scripts/fetch_qqp_sample.py --hit-count 300 --miss-count 300
"""

import argparse
import json
import os
import urllib.error
import urllib.request

DATASET = "AlekseyKorshuk/quora-question-pairs"
API_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
MAX_OFFSET = 20000  # safety cap so a bad quota doesn't page through the whole dataset

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "qqp_pairs.json")


def fetch_page(offset, length=PAGE_SIZE):
    """Fetch one page of rows from the QQP dataset via datasets-server."""
    url = (
        f"{API_URL}?dataset={DATASET}&config=default&split=train"
        f"&offset={offset}&length={length}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch QQP page at offset={offset}: {e}") from e


def sample_pairs(hit_count, miss_count):
    """Page through QQP until `hit_count` duplicate and `miss_count`
    non-duplicate pairs are collected, or MAX_OFFSET is reached."""
    hits, misses = [], []
    offset = 0

    while (len(hits) < hit_count or len(misses) < miss_count) and offset < MAX_OFFSET:
        rows = fetch_page(offset).get("rows", [])
        if not rows:
            break

        for entry in rows:
            row = entry["row"]
            q1, q2 = row["question1"], row["question2"]
            if not q1 or not q2 or not q1.strip() or not q2.strip():
                continue

            if row["is_duplicate"] and len(hits) < hit_count:
                hits.append({"a": q1, "b": q2, "label": "hit"})
            elif not row["is_duplicate"] and len(misses) < miss_count:
                misses.append({"a": q1, "b": q2, "label": "miss"})

        offset += PAGE_SIZE

    return hits, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hit-count", type=int, default=200, help="Number of duplicate (should-hit) pairs to sample.")
    parser.add_argument("--miss-count", type=int, default=200, help="Number of non-duplicate (should-miss) pairs to sample.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path.")
    args = parser.parse_args()

    hits, misses = sample_pairs(args.hit_count, args.miss_count)
    if len(hits) < args.hit_count or len(misses) < args.miss_count:
        print(f"Warning: only found {len(hits)}/{args.hit_count} hit pairs and "
              f"{len(misses)}/{args.miss_count} miss pairs before MAX_OFFSET={MAX_OFFSET}.")

    pairs = hits + misses
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} pairs ({len(hits)} hit, {len(misses)} miss) to {out_path}")


if __name__ == "__main__":
    main()
