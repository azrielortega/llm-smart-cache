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
question ─▶ Embedder ─▶ VectorDB.search (FAISS, L2 distance)
                              │
                distance < CACHE_MAX_DISTANCE?
                    │                    │
                   yes                  no
                    │                    │
              return cached          call the LLM (OpenRouter),
              answer, no LLM         cache the new answer,
              call                   return it
```

- **`core/embedder.py`**: wraps `SentenceTransformer`, turns text into vectors.
- **`core/vector_db.py`**: owns the FAISS index plus per-entry metadata
  (question, answer, timestamps) as one unit, with TTL and LRU/max-size
  eviction so the index doesn't grow unbounded. Persists to disk
  (`index.faiss` + `metadata.json`).
- **`core/cache_logic.py`** (`SmartCache`): the public interface, `query()` /
  `update()` / `save()`, combining the embedder and vector DB.
- **`core/llm_client.py`**: thin wrapper around the OpenAI SDK, pointed at
  [OpenRouter](https://openrouter.ai/), used only on a cache miss.
- **`core/config.py`**: every tunable (model names, distance threshold,
  pricing, log level) reads from the environment with a working default.

## Setup

Requires Python 3.10+.

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
from core.llm_client import build_client, call_llm

cache = SmartCache()
client = build_client()

question = "How do I bake a cake?"
answer = cache.query(question)
if answer is None:
    completion = call_llm(client, question)
    answer = completion.choices[0].message.content
    cache.update(question, answer)

cache.save()
```

## Configuration

All read from the environment (or a `.env` file); every value has a working
default, so none of this is required.

| Variable                      | Default                | Meaning                                                                 |
|--------------------------------|------------------------|--------------------------------------------------------------------------|
| `OPENROUTER_KEY`               | none                    | API key for real LLM calls. Required unless you only use `--mock`.       |
| `EMBEDDING_MODEL_NAME`         | `all-MiniLM-L6-v2`      | `sentence-transformers` model used to embed questions.                   |
| `EMBEDDING_DIMENSION`          | `384`                   | Must match the embedding model's output dimension.                       |
| `CACHE_MAX_DISTANCE`           | `0.56`                  | Max squared L2 distance for a cache hit. Lower = stricter matching. Depends on `EMBEDDING_MODEL_NAME` - see [Tuning the threshold](#tuning-the-threshold). |
| `LLM_MODEL_NAME`                | `openai/gpt-4o-mini`    | OpenRouter model id used on a cache miss.                                |
| `LOG_LEVEL`                     | `INFO`                  | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                                   |
| `LLM_PROMPT_COST_PER_1K`        | `0.00015`               | USD/1K prompt tokens, used by `benchmark.py` to estimate $ saved.        |
| `LLM_COMPLETION_COST_PER_1K`    | `0.0006`                | USD/1K completion tokens, used by `benchmark.py` to estimate $ saved.    |

`SmartCache` also takes constructor overrides (`max_distance`, `cache_dir`,
`ttl_seconds`, `max_size`, `model_name`, `embedding_dimension`) for anything
you don't want to set globally via env var.

### Tuning the threshold

`CACHE_MAX_DISTANCE` is a property of `EMBEDDING_MODEL_NAME`, not a universal
constant - the `0.56` default was picked by sweeping candidate thresholds
against a labeled sample of real question pairs and choosing the best
precision/recall balance for `all-MiniLM-L6-v2`. See [`eval/README.md`](eval/README.md)
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
  cache_logic.py    SmartCache - query / update / save
  vector_db.py       FAISS index + metadata + TTL/LRU eviction
  embedder.py         sentence-transformers wrapper
  llm_client.py       OpenRouter client + call_llm()
  mock_llm.py          fake client for --mock
  logging_config.py    setup_logging()
  config.py             env-var settings
cli.py                interactive REPL / single-shot demo
benchmark.py           cached vs. uncached comparison
eval/
  scripts/
    fetch_qqp_sample.py  one-off: samples labeled pairs into eval/data/
  data/                   qqp_pairs.json - committed eval fixture
  tune_threshold.py      sweeps CACHE_MAX_DISTANCE, reports precision/recall/F1
main.py                minimal scripted example
tests/                 pytest suite
```
