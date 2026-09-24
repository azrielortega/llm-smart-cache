"""Interactive CLI for llm-smart-cache - ask questions and see whether they
hit the cache, without reading any source.

Usage:
    python cli.py                  # interactive REPL, real OpenRouter calls
    python cli.py --mock           # interactive REPL, simulated LLM (no API key/network)
    python cli.py "question"       # single-shot: ask one question, print the answer, exit
    python cli.py "question" --mock

In the REPL, `:stats` prints this session's hit rate and `:quit` (or Ctrl-D) exits.
"""

import argparse
import logging
import time

from core.cache_logic import SmartCache
from core.llm_client import build_client, get_or_call
from core.logging_config import setup_logging
from core.mock_llm import MockClient

logger = logging.getLogger(__name__)

QUIT_COMMANDS = (":quit", ":exit", ":q")
STATS_COMMANDS = (":stats", ":s")


def ask(cache, client, question):
    """Query the cache, falling back to the LLM on a miss.

    Returns (answer, hit, latency_seconds).
    """
    start = time.perf_counter()
    answer, completion = get_or_call(cache, client, question)
    return answer, completion is None, time.perf_counter() - start


def print_result(answer, hit, latency):
    badge = "HIT " if hit else "MISS"
    print(f"[{badge} {latency * 1000:.0f}ms] {answer}")


def print_stats(hits, total):
    if total == 0:
        print("No questions asked yet this session.")
        return
    print(f"Session: {total} question(s), {hits} cache hit(s) ({hits / total:.0%} hit rate).")


def run_repl(cache, client):
    print("llm-smart-cache - ask a question. `:stats` for session stats, `:quit` to exit.\n")
    hits = 0
    total = 0
    try:
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                print()
                break

            if not question:
                continue
            if question in QUIT_COMMANDS:
                break
            if question in STATS_COMMANDS:
                print_stats(hits, total)
                continue

            try:
                answer, hit, latency = ask(cache, client, question)
            except (ValueError, RuntimeError) as e:
                print(f"Error: {e}")
                continue

            total += 1
            hits += int(hit)
            print_result(answer, hit, latency)
    finally:
        print_stats(hits, total)
        cache.save()


def run_once(cache, client, question):
    try:
        answer, hit, latency = ask(cache, client, question)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return
    print_result(answer, hit, latency)
    cache.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="Ask a single question and exit instead of starting the REPL.")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a simulated LLM instead of real OpenRouter calls (no API key or network needed).",
    )
    args = parser.parse_args()

    setup_logging()

    client = MockClient() if args.mock else build_client()
    cache = SmartCache()

    if args.question:
        run_once(cache, client, args.question)
    else:
        run_repl(cache, client)


if __name__ == "__main__":
    main()
