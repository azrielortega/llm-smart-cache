"""Benchmark: runs the same set of user queries through an uncached baseline
(every query calls the LLM) and through SmartCache (repeats/paraphrases hit
the cache instead), then reports hit rate, latency, and estimated $ saved.

Usage:
    python benchmark.py            # real OpenRouter calls, needs OPENROUTER_KEY
    python benchmark.py --mock     # simulated LLM, no API key or network needed
"""

import argparse
import logging
import tempfile
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from core.cache_logic import SmartCache
from core.config import LLM_COMPLETION_COST_PER_1K, LLM_MODEL_NAME, LLM_PROMPT_COST_PER_1K
from core.llm_client import build_client, call_llm, get_or_call
from core.logging_config import setup_logging
from core.mock_llm import MockClient

logger = logging.getLogger(__name__)

# A simulated session of user questions: each group is one topic asked a few
# different ways, so the cached pass sees a realistic mix of semantic hits
# (paraphrases of an already-asked question) and misses (a genuinely new one).
QUERY_GROUPS = [
    [
        "How do I bake a chocolate cake?",
        "How to bake a chocolate cake?",
        "What's the process for baking a chocolate cake?",
    ],
    [
        "What's the capital of France?",
        "What is France's capital city?",
    ],
    [
        "How do I reverse a linked list in Python?",
        "How to reverse a linked list in Python?",
    ],
    [
        "What's a good beginner workout routine?",
    ],
    [
        "Explain how photosynthesis works.",
        "Can you explain photosynthesis?",
    ],
    [
        "What's the best way to learn a new language?",
    ],
    [
        "What are the tradeoffs between REST and GraphQL for a mobile app "
        "backend, and which would you recommend for a team building an "
        "offline-first app?",
        "Comparing REST and GraphQL for a mobile backend - what are the "
        "tradeoffs, and what's better for an offline-first app?",
    ],
    [
        "What's the difference between SQL and NoSQL databases, and when "
        "should I pick one over the other for a high-write analytics workload?",
        "SQL vs NoSQL for a high-write analytics workload - what's the "
        "difference, and which should I use?",
    ],
    [
        "How would you design a rate limiter for a public API that needs to "
        "support both per-user and per-IP limits, including how it should "
        "behave under a traffic spike?",
        "Walk me through designing a rate limiter for a public API with "
        "per-user and per-IP limits that also holds up under a traffic spike.",
    ],
    [
        "What are the pros and cons of a microservices architecture versus a "
        "monolith for a team of about five engineers?",
    ],
]

QUERIES = [query for group in QUERY_GROUPS for query in group]


@dataclass
class CallResult:
    latency: float
    hit: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


def estimate_cost(result):
    return (
        result.prompt_tokens / 1000 * LLM_PROMPT_COST_PER_1K
        + result.completion_tokens / 1000 * LLM_COMPLETION_COST_PER_1K
    )


def run_uncached(client, queries):
    """Baseline: call the LLM for every query, with no caching at all."""
    results = []
    for query in queries:
        start = time.perf_counter()
        completion = call_llm(client, query)
        latency = time.perf_counter() - start
        results.append(CallResult(
            latency=latency,
            hit=False,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        ))
    return results


def run_cached(client, cache, queries):
    """Same queries through SmartCache: a hit skips the LLM call entirely."""
    results = []
    for query in queries:
        start = time.perf_counter()
        _, completion = get_or_call(cache, client, query)
        latency = time.perf_counter() - start
        if completion is None:
            results.append(CallResult(latency=latency, hit=True))
            continue

        results.append(CallResult(
            latency=latency,
            hit=False,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        ))
    return results


def summarize(results):
    total_calls = len(results)
    hits = sum(1 for r in results if r.hit)
    total_latency = sum(r.latency for r in results)
    return {
        "total_calls": total_calls,
        "hits": hits,
        "hit_rate": hits / total_calls if total_calls else 0.0,
        "total_latency": total_latency,
        "avg_latency": total_latency / total_calls if total_calls else 0.0,
        "total_cost": sum(estimate_cost(r) for r in results),
    }


def print_report(uncached, cached):
    saved_latency = uncached["total_latency"] - cached["total_latency"]
    saved_cost = uncached["total_cost"] - cached["total_cost"]

    rows = [
        ("Queries", str(uncached["total_calls"]), str(cached["total_calls"])),
        ("Cache hit rate", "n/a", f"{cached['hit_rate']:.0%}"),
        ("Total latency", f"{uncached['total_latency']:.2f}s", f"{cached['total_latency']:.2f}s"),
        ("Avg latency/query", f"{uncached['avg_latency'] * 1000:.0f}ms", f"{cached['avg_latency'] * 1000:.0f}ms"),
        ("Estimated cost", f"${uncached['total_cost']:.5f}", f"${cached['total_cost']:.5f}"),
    ]

    label_w, col_w = 20, 15
    print()
    print("=" * (label_w + 2 * col_w))
    print(f"  llm-smart-cache benchmark  (model={LLM_MODEL_NAME})")
    print("=" * (label_w + 2 * col_w))
    print(f"{'':<{label_w}}{'Uncached':>{col_w}}{'Cached':>{col_w}}")
    for label, uncached_val, cached_val in rows:
        print(f"{label:<{label_w}}{uncached_val:>{col_w}}{cached_val:>{col_w}}")
    print("-" * (label_w + 2 * col_w))

    if uncached["total_latency"] > 0:
        print(f"  Latency saved:     {saved_latency:.2f}s "
              f"({saved_latency / uncached['total_latency']:.0%} faster)")
    if uncached["total_cost"] > 0:
        print(f"  Estimated $ saved: ${saved_cost:.5f} "
              f"({saved_cost / uncached['total_cost']:.0%} cheaper)")
    print("=" * (label_w + 2 * col_w))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a simulated LLM instead of real OpenRouter calls (no API key or network needed).",
    )
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    client = MockClient() if args.mock else build_client()

    with tempfile.TemporaryDirectory() as cache_dir:
        cache = SmartCache(cache_dir=cache_dir)

        logger.info("Running uncached baseline (%d queries)...", len(QUERIES))
        uncached_results = run_uncached(client, QUERIES)

        logger.info("Running cached pass (%d queries)...", len(QUERIES))
        cached_results = run_cached(client, cache, QUERIES)

    print_report(summarize(uncached_results), summarize(cached_results))


if __name__ == "__main__":
    main()
