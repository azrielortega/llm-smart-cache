from core.mock_llm import MockClient


def test_mock_completion_has_answer_and_token_usage(monkeypatch):
    monkeypatch.setattr("core.mock_llm.time.sleep", lambda _: None)

    completion = MockClient().chat.completions.create(
        model="any-model",
        messages=[{"role": "user", "content": "How do I bake a cake?"}],
    )

    assert "How do I bake a cake?" in completion.choices[0].message.content
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0
