"""A fake OpenAI-shaped client, so `benchmark.py` and `cli.py` can be run with
`--mock` - no OPENROUTER_KEY or network access needed to see how the project
behaves."""

import random
import time


class MockClient:
    """Stands in for the OpenAI client. Simulates realistic network latency and
    returns a canned answer with plausible token counts instead of a real
    completion."""

    class _Usage:
        def __init__(self, prompt_tokens, completion_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = MockClient._Message(content)

    class _Completion:
        def __init__(self, content, prompt_tokens, completion_tokens):
            self.choices = [MockClient._Choice(content)]
            self.usage = MockClient._Usage(prompt_tokens, completion_tokens)

    class _Completions:
        def create(self, model, messages):
            time.sleep(random.uniform(0.3, 0.8))  # simulate real network latency
            question = messages[0]["content"]
            answer = f"[mock answer to: {question}]"
            return MockClient._Completion(
                answer,
                prompt_tokens=max(1, len(question.split())),
                completion_tokens=max(1, len(answer.split())),
            )

    class _Chat:
        def __init__(self):
            self.completions = MockClient._Completions()

    def __init__(self):
        self.chat = MockClient._Chat()
