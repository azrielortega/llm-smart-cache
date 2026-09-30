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

def _get_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        number = float(value)

        if not number.is_integer():
            raise ValueError(f"Env var {name}={value!r} is not a valid integer")

        return number
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid integer")


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CACHE_MAX_DISTANCE = _get_float("CACHE_MAX_DISTANCE", 0.25)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
LLM_TIMEOUT_S = _get_float("LLM_TIMEOUT_S", 30)
LLM_MAX_RETRIES = _get_int("LLM_MAX_RETRIES", 3)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Rough OpenRouter pricing (USD per 1K tokens) for the default LLM_MODEL_NAME
# ("openai/gpt-4o-mini"), used only to estimate $ saved in benchmark.py. If you
# change LLM_MODEL_NAME to a model with different pricing, override these.
LLM_PROMPT_COST_PER_1K = _get_float("LLM_PROMPT_COST_PER_1K", 0.00015)
LLM_COMPLETION_COST_PER_1K = _get_float("LLM_COMPLETION_COST_PER_1K", 0.0006)
