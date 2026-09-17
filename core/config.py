"""Central place for env-configurable settings.

Everything here has a sane default, so nothing needs to be set for the
project to run - set the corresponding env var (or put it in `.env`) to
override.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid int")


def _get_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Env var {name}={value!r} is not a valid float")


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = _get_int("EMBEDDING_DIMENSION", 384)
CACHE_MAX_DISTANCE = _get_float("CACHE_MAX_DISTANCE", 0.3)
