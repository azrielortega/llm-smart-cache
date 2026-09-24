import pytest

from core.config import _get_float


def test_get_float_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert _get_float("TEST_FLOAT", 0.5) == 0.5


def test_get_float_parses_env_value(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0.42")
    assert _get_float("TEST_FLOAT", 0.5) == 0.42


def test_get_float_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "abc")
    with pytest.raises(ValueError, match="TEST_FLOAT='abc' is not a valid float"):
        _get_float("TEST_FLOAT", 0.5)
