"""Central place for env-configurable settings.

Everything here has a sane default, so nothing needs to be set for the
project to run - set the corresponding env var (or put it in `.env`) to
override.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid float")


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CACHE_MAX_DISTANCE = _get_float("CACHE_MAX_DISTANCE", 0.56)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Rough OpenRouter pricing (USD per 1K tokens) for the default LLM_MODEL_NAME
# ("openai/gpt-4o-mini"), used only to estimate $ saved in benchmark.py. If you
# change LLM_MODEL_NAME to a model with different pricing, override these.
LLM_PROMPT_COST_PER_1K = _get_float("LLM_PROMPT_COST_PER_1K", 0.00015)
LLM_COMPLETION_COST_PER_1K = _get_float("LLM_COMPLETION_COST_PER_1K", 0.0006)
