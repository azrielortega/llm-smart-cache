# llm-smart-cache

A semantic cache for LLM calls. Incoming questions are embedded with
[`sentence-transformers`](https://www.sbert.net/) and checked against a
[FAISS](https://github.com/facebookresearch/faiss) index of previously answered
questions; a close-enough match returns the cached answer instead of paying for
another LLM call. Paraphrases hit the cache too, not just exact repeats.

### Why this matters

LLM calls are slow and billed per token, but real traffic is full of
repeats: the same handful of questions asked in slightly different words,
over and over (support bots, FAQ assistants, internal tools). An exact-match
cache misses almost all of that because it can't tell "how do I bake a cake"
from "how to bake a cake." A semantic cache catches both, which is what
directly cuts latency and LLM spend at scale instead of shaving off a sliver
of exact-duplicate traffic.

```
> How do I bake a chocolate cake?
[MISS 812ms] Preheat the oven to 350°F, cream the butter and sugar...

> What's the process for baking a chocolate cake?
[HIT  9ms] Preheat the oven to 350°F, cream the butter and sugar...
```

## How it works

```
question ─▶ Embedder ─▶ VectorDB.search (FAISS, top search_k neighbors,
                              │         TTL-expired ones skipped)
                              │
          any neighbor with distance < CACHE_MAX_DISTANCE?
                    │                    │
                   yes                  no
                    │                    │
              return the closest     call the LLM (OpenRouter),
              one's cached answer,   cache the new answer,
              no LLM call            return it
```

Checking the top `search_k` neighbors (default 5) instead of only the nearest
one matters when TTL is enabled: if the closest match has expired, a slightly
further but still fresh match can still produce a hit.

- **`core/embedder.py`**: wraps `SentenceTransformer`, turns text into vectors.
- **`core/vector_db.py`**: owns the FAISS index plus per-entry metadata
  (question, answer, timestamps) as one unit, with TTL and LRU/max-size
  eviction so the index doesn't grow unbounded. Persists to disk
  (`index.faiss` + `metadata.json`).
- **`core/cache_logic.py`** (`SmartCache`): the public interface, `query()` /
  `update()` / `save()`, combining the embedder and vector DB. `query()` returns
  the answer of the closest fresh neighbor under the threshold, or `None` on a miss.
  It also records the embedding model in `embedding_model.txt` and the LLM
  model in `llm_model.txt`, and refuses to load a non-empty cache built with a
  different one: vectors from different embedding models aren't comparable,
  and answers from a different LLM would be silently stale.
- **`core/llm_client.py`**: thin wrapper around the OpenAI SDK, pointed at
  [OpenRouter](https://openrouter.ai/), used only on a cache miss.
- **`core/config.py`**: every tunable (model names, distance threshold,
  pricing, log level) reads from the environment with a working default.

## Setup

Requires Python 3.12+ (the pinned `numpy>=2.5.3` needs it).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENROUTER_KEY (get one at https://openrouter.ai/keys)
```

`OPENROUTER_KEY` is only required for real LLM calls (`main.py`, or `cli.py` /
`benchmark.py` without `--mock`). Everything else has a sane default: see
[Configuration](#configuration).

## Usage

### CLI: try it interactively

```bash
python cli.py                  # interactive REPL, real OpenRouter calls
python cli.py --mock           # interactive REPL, simulated LLM (no API key/network needed)
python cli.py "question"       # single-shot: ask one question, print the answer, exit
```

In the REPL, `:stats` prints the session's hit rate and `:quit` (or Ctrl-D) exits.

### Benchmark: quantify the value

Runs the same set of questions once with no cache (every question hits the
LLM) and once through `SmartCache` (paraphrases hit the cache instead), and
reports the difference:

```bash
python benchmark.py            # real OpenRouter calls, needs OPENROUTER_KEY
python benchmark.py --mock     # simulated LLM, no API key or network needed
```

```
==================================================
  llm-smart-cache benchmark  (model=openai/gpt-4o-mini)
==================================================
                           Uncached         Cached
Queries                          18             18
Cache hit rate                  n/a            44%
Total latency                10.87s          6.54s
Avg latency/query             604ms          363ms
Estimated cost             $0.00020       $0.00012
--------------------------------------------------
  Latency saved:     4.33s (40% faster)
  Estimated $ saved: $0.00008 (41% cheaper)
==================================================
```

($ saved is estimated from real token usage on the calls that do go through,
multiplied by `LLM_PROMPT_COST_PER_1K` / `LLM_COMPLETION_COST_PER_1K`: see
Configuration.)

### Library usage

```python
from core.cache_logic import SmartCache
from core.llm_client import build_client, get_or_call

cache = SmartCache()
client = build_client()

# completion is None on a cache hit, or the raw LLM completion on a miss.
answer, completion = get_or_call(cache, client, "How do I bake a cake?")

cache.save()
```

## Configuration

All read from the environment (or a `.env` file); every value has a working
default, so none of this is required.

| Variable                      | Default                | Meaning                                                                 |
|--------------------------------|------------------------|--------------------------------------------------------------------------|
| `OPENROUTER_KEY`               | none                    | API key for real LLM calls. Required unless you only use `--mock`.       |
| `EMBEDDING_MODEL_NAME`         | `all-MiniLM-L6-v2`      | `sentence-transformers` model used to embed questions. Changing it needs a fresh `cache_dir` (or delete `cache_data/`). |
| `CACHE_MAX_DISTANCE`           | `0.56`                  | Max squared L2 distance for a cache hit. Lower = stricter matching. Depends on `EMBEDDING_MODEL_NAME`, see [Tuning the threshold](#tuning-the-threshold). |
| `LLM_MODEL_NAME`                | `openai/gpt-4o-mini`    | OpenRouter model id used on a cache miss. Changing it needs a fresh `cache_dir` (or delete `cache_data/`). |
| `LOG_LEVEL`                     | `INFO`                  | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                                   |
| `LLM_PROMPT_COST_PER_1K`        | `0.00015`               | USD/1K prompt tokens, used by `benchmark.py` to estimate $ saved.        |
| `LLM_COMPLETION_COST_PER_1K`    | `0.0006`                | USD/1K completion tokens, used by `benchmark.py` to estimate $ saved.    |

`SmartCache` also takes constructor overrides for anything you don't want to
set globally via env var:

| Argument              | Default                  | Meaning                                                              |
|-----------------------|--------------------------|----------------------------------------------------------------------|
| `max_distance`        | `CACHE_MAX_DISTANCE`     | Hit threshold (squared L2 distance).                                 |
| `cache_dir`           | `cache_data`             | Where `index.faiss`, `metadata.json`, `embedding_model.txt` and `llm_model.txt` are stored. |
| `ttl_seconds`         | `None` (no expiry)       | Entries older than this are ignored on search and dropped on the next insert. |
| `max_size`            | `1000`                   | Max entries kept; least recently used ones are evicted past this.   |
| `search_k`            | `5`                      | Nearest neighbors checked per query. Only matters when `ttl_seconds` is set. |
| `model_name`          | `EMBEDDING_MODEL_NAME`   | Embedding model.                                                     |
| `embedder`            | `None`                   | Ready-made `Embedder` to use instead of loading `model_name` (e.g. a test fake). |
| `llm_model_name`      | `LLM_MODEL_NAME`         | LLM the cached answers come from. Changing it needs a fresh `cache_dir`. |

### Tuning the threshold

`CACHE_MAX_DISTANCE` is a property of `EMBEDDING_MODEL_NAME`, not a universal
constant. The `0.56` default was picked by sweeping candidate thresholds
against a labeled sample of Quora question pairs and taking the one with the
best F1 score for `all-MiniLM-L6-v2` (72.4% precision, 86.5% recall). See [`eval/README.md`](eval/README.md)
for the methodology, results, and how to re-tune it if you change the model.

## Testing

```bash
pytest
```

Tests stub out `sentence-transformers` with a deterministic, hash-seeded fake
embedder (`tests/conftest.py`), so the suite runs without downloading a model
or needing `torch`. `faiss` and `numpy` are used for real in the `VectorDB`
tests.

## Project structure

```
core/
  cache_logic.py    SmartCache: query / update / save
  vector_db.py       FAISS index + metadata + TTL/LRU eviction
  embedder.py         sentence-transformers wrapper
  llm_client.py       OpenRouter client + call_llm() + get_or_call()
  mock_llm.py          fake client for --mock
  logging_config.py    setup_logging()
  config.py             env-var settings
cli.py                interactive REPL / single-shot demo
benchmark.py           cached vs. uncached comparison
eval/
  scripts/
    fetch_qqp_sample.py  one-off: samples labeled pairs into eval/data/
  data/                   qqp_pairs.json: committed eval fixture
  tune_threshold.py      sweeps CACHE_MAX_DISTANCE, reports precision/recall/F1
main.py                minimal scripted example
tests/                 pytest suite
```
