"""Central place for env-configurable settings.

Everything here has a sane default, so nothing needs to be set for the
project to run - set the corresponding env var (or put it in `.env`) to
override.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_float(name, default, positive=False):
    """Read a float env var, or return default when unset.

    Inputs:  name (str), default (float), positive (bool) - if True, reject values <= 0
    Outputs: float; raises ValueError on a non-float or out-of-range value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid float")

    if positive and number <= 0:
        raise ValueError(f"Env var {name}={value!r} must be greater than 0")

    return number

def _get_int(name, default, non_negative=False):
    """Read an integer env var, or return default when unset. Floats like "5.0" are rejected.

    Inputs:  name (str), default (int), non_negative (bool) - if True, reject values < 0
    Outputs: int; raises ValueError on a non-integer or out-of-range value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid integer")

    if non_negative and number < 0:
        raise ValueError(f"Env var {name}={value!r} must be 0 or greater")

    return number


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CACHE_MAX_DISTANCE = _get_float("CACHE_MAX_DISTANCE", 0.25)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")
LLM_TIMEOUT_S = _get_float("LLM_TIMEOUT_S", 30, positive=True)
LLM_MAX_RETRIES = _get_int("LLM_MAX_RETRIES", 3, non_negative=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Rough OpenRouter pricing (USD per 1K tokens) for the default LLM_MODEL_NAME
# ("openai/gpt-4o-mini"), used only to estimate $ saved in benchmark.py. If you
# change LLM_MODEL_NAME to a model with different pricing, override these.
LLM_PROMPT_COST_PER_1K = _get_float("LLM_PROMPT_COST_PER_1K", 0.00015)
LLM_COMPLETION_COST_PER_1K = _get_float("LLM_COMPLETION_COST_PER_1K", 0.0006)
