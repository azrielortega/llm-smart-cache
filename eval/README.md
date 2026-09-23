# Threshold evaluation

Tools for picking a defensible `CACHE_MAX_DISTANCE` (see `core/config.py`)
by testing candidate values against labeled real question pairs, instead of
guessing a number.

## Why this exists

`SmartCache` checks the top `search_k` nearest stored questions (FAISS
`IndexFlatL2`) and calls it a hit if any fresh one is closer than
`CACHE_MAX_DISTANCE`. That number is a property of `EMBEDDING_MODEL_NAME`,
not a universal constant: different embedding models spread the same
paraphrase / non-paraphrase pairs across different distance ranges. Change
the model and the old threshold has no guaranteed relationship to the new
distance scale.

## Dataset

[Quora Question Pairs](https://huggingface.co/datasets/AlekseyKorshuk/quora-question-pairs):
a public, human-labeled corpus of ~404K question pairs marked duplicate /
not-duplicate. `is_duplicate=1` → a pair the cache *should* hit on
(paraphrase); `is_duplicate=0` → a pair it *should* miss on (different
question).

## Usage

```bash
# One-off: samples labeled pairs from QQP via Hugging Face's datasets-server
# API (no auth, no extra dependencies) and writes eval/data/qqp_pairs.json.
python eval/scripts/fetch_qqp_sample.py
python eval/scripts/fetch_qqp_sample.py --hit-count 300 --miss-count 300

# Embeds every pair, sweeps candidate thresholds, reports precision/recall/F1.
python -m eval.tune_threshold
python -m eval.tune_threshold --min-precision 0.85
```

`eval/data/qqp_pairs.json` is committed, so `tune_threshold.py` needs no
network access by default. Only re-run the fetch script if you want a
different or larger sample.

## How it works

1. **`scripts/fetch_qqp_sample.py`** pages through QQP, balances hit/miss
   pairs, and writes them as `[{"a": q1, "b": q2, "label": "hit"|"miss"}, ...]`.
2. **`tune_threshold.py`** embeds every unique question once via
   `core.embedder.Embedder` (the same code path `SmartCache` uses),
   computes squared L2 distance per pair (matching FAISS `IndexFlatL2`'s
   metric), sweeps candidate thresholds across the observed distance range,
   and scores each one against the labels (precision / recall / F1).
3. Below the table, the script also prints a "Recommended" threshold: the
   one with the highest recall that still keeps precision at or above
   `--min-precision` (default `0.95`). With `all-MiniLM-L6-v2` no threshold
   reaches 95%, so the default run prints "No threshold reaches..." and you
   pick from the table instead, or pass a lower `--min-precision`.

## Current results (`all-MiniLM-L6-v2`)

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.249 | 86.2% | 47.0% | 60.8% |
| 0.311 (old default) | 83.9% | 57.5% | 68.2% |
| 0.436 | 78.2% | 75.5% | 76.8% |
| **0.561 (current default, rounded to 0.56)** | **72.4%** | **86.5%** | **78.8% ← best F1** |
| 0.685 | 66.1% | 93.5% | 77.4% |
| 0.872+ | ~61% and falling | 100.0% | falling |

Precision tops out around 88-91%, and only at very low recall (~12-21%).
That's a real limit of this model's semantic separation on hard pairs, not a
bug. The default `CACHE_MAX_DISTANCE=0.56` was read off this table as the
best F1 score, favoring catching more true paraphrases over minimizing wrong
cache hits.

## Re-tuning after a model change

If `EMBEDDING_MODEL_NAME` changes, re-run `python -m eval.tune_threshold`
and update `CACHE_MAX_DISTANCE` (in `.env` or the default in
`core/config.py`) to match the new sweep. Do not carry the old value over.
