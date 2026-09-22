"""Sweeps candidate CACHE_MAX_DISTANCE values against a labeled set of Quora
question pairs (see eval/scripts/fetch_qqp_sample.py) and reports precision/
recall/F1 per threshold, to pick a defensible value instead of guessing one.

Usage (run from the repo root):
    python -m eval.tune_threshold
    python -m eval.tune_threshold --min-precision 0.9
"""

import argparse
import json
import os

import numpy as np

from core.embedder import Embedder

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "qqp_pairs.json")
NUM_STEPS = 40

def squared_l2(va, vb):
    """FAISS IndexFlatL2 returns *squared* L2 distance - match that here so
    thresholds reported are directly comparable to CACHE_MAX_DISTANCE."""
    return float(np.sum((va - vb) ** 2))

def load_pairs(path):
    with open(path) as f:
        pairs = json.load(f)
    if not pairs:
        raise ValueError(f"No pairs found in {path!r} - run eval/scripts/fetch_qqp_sample.py first.")
    return pairs

def compute_distances(embedder, pairs):
    """Embed every unique text once, then return (distance, is_hit) per pair."""
    cache = {}

    def encode(text):
        if text not in cache:
            cache[text] = embedder.encode(text)[0]
        return cache[text]

    return [(squared_l2(encode(p["a"]), encode(p["b"])), p["label"] == "hit") for p in pairs]

def metrics_at(distances, threshold):
    tp = fp = fn = tn = 0
    for distance, is_hit in distances:
        predicted_hit = distance < threshold
        if predicted_hit and is_hit:
            tp += 1
        elif predicted_hit and not is_hit:
            fp += 1
        elif not predicted_hit and is_hit:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def sweep(distances, num_steps=NUM_STEPS):
    """Sweep thresholds across the observed distance range, rather than a
    fixed grid - the embedding model sets the real scale, not us."""
    hi = max(d for d, _ in distances) * 1.05
    step = hi / num_steps
    thresholds = [round(step * i, 4) for i in range(1, num_steps + 1)]
    return [metrics_at(distances, t) for t in thresholds]

def recommend(results, min_precision):
    """Highest (loosest) threshold that keeps precision >= min_precision - a
    false hit (wrong cached answer) is worse than a false miss (one extra LLM
    call), so precision is prioritized over recall."""
    candidates = [r for r in results if r["precision"] >= min_precision]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["recall"], r["threshold"]))

def print_report(results, recommended, min_precision):
    print()
    print("=" * 60)
    print("  threshold tuning  (squared L2 distance)")
    print("=" * 60)
    print(f"{'threshold':>10}{'precision':>12}{'recall':>10}{'f1':>8}{'hits':>8}{'misses':>8}")
    for r in results:
        print(f"{r['threshold']:>10.3f}{r['precision']:>12.1%}{r['recall']:>10.1%}"
              f"{r['f1']:>8.1%}{r['tp'] + r['fn']:>8}{r['fp'] + r['tn']:>8}")
    print("-" * 60)
    if recommended:
        print(f"  Recommended CACHE_MAX_DISTANCE = {recommended['threshold']:.3f} "
              f"(precision={recommended['precision']:.1%}, recall={recommended['recall']:.1%}, "
              f"min_precision={min_precision:.0%})")
    else:
        print(f"  No threshold reaches precision >= {min_precision:.0%} - try a lower --min-precision.")
    print("=" * 60)
    print()

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to labeled pairs JSON.")
    parser.add_argument("--min-precision", type=float, default=0.95,
                         help="Minimum precision the recommended threshold must keep (default 0.95).")
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    embedder = Embedder()
    distances = compute_distances(embedder, pairs)

    results = sweep(distances)
    recommended = recommend(results, args.min_precision)
    print_report(results, recommended, args.min_precision)

if __name__ == "__main__":
    main()
