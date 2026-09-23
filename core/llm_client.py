"""OpenAI client pointed at OpenRouter, shared by main.py, cli.py and benchmark.py."""

import logging
import os

from openai import OpenAI

from core.config import LLM_MODEL_NAME

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def build_client():
    """Build an OpenAI client for OpenRouter using OPENROUTER_KEY.

    Outputs: OpenAI - configured client; raises ValueError if the key is missing.
    """
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        raise ValueError("API KEY not found! Set OPENROUTER_KEY in your .env file.")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def call_llm(client, user_input, model=None):
    """Send one user message to the LLM.

    Inputs:  client (OpenAI | MockClient), user_input (str), model (str | None) - defaults to LLM_MODEL_NAME
    Outputs: ChatCompletion - raw completion; raises RuntimeError on failure.
    """
    model = model or LLM_MODEL_NAME
    logger.debug("Calling LLM API (model=%s) for: %r", model, user_input)
    try:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed for model {model!r}: {e}") from e
