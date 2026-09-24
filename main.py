import logging

from core.cache_logic import SmartCache
from core.llm_client import build_client, get_or_call
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_response(cache, client, user_input):
    answer, _ = get_or_call(cache, client, user_input)
    return answer


if __name__ == "__main__":
    setup_logging()

    cache = SmartCache()
    client = build_client()

    print(get_response(cache, client, "How to bake a cake?"))     # MISS -> calls the LLM
    print(get_response(cache, client, "How do I bake a cake?"))   # HIT  -> semantic match, no LLM call
    cache.save()
