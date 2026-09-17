import logging

from dotenv import load_dotenv

from core.cache_logic import SmartCache
from core.llm_client import build_client, call_llm
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()


def get_response(cache, client, user_input):
    cached_answer = cache.query(user_input)
    if cached_answer:
        return cached_answer

    completion = call_llm(client, user_input)
    new_answer = completion.choices[0].message.content
    cache.update(user_input, new_answer)
    return new_answer


if __name__ == "__main__":
    setup_logging()

    cache = SmartCache()
    client = build_client()

    print(get_response(cache, client, "How to bake a cake?"))     # MISS -> calls the LLM
    print(get_response(cache, client, "How do I bake a cake?"))   # HIT  -> semantic match, no LLM call
    cache.save()
