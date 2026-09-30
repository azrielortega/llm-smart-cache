import pytest

from core.config import _get_float, _get_int

#============================= FLOAT RETRIEVAL ====================================

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

#============================= INTEGER RETRIEVAL ====================================

def test_get_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert _get_int("TEST_INT", 0.5) == 0.5


def test_get_int_parses_env_value(monkeypatch):
    monkeypatch.setenv("TEST_INT", "67")
    assert _get_int("TEST_INT", 67) == 67


def test_get_int_parses_env_value_2(monkeypatch):
    monkeypatch.setenv("TEST_INT", "5.0")
    assert _get_int("TEST_INT", 5.0) == 5


def test_get_int_rejects_invalid_value_str(monkeypatch):
    monkeypatch.setenv("TEST_INT", "abc")
    with pytest.raises(ValueError, match="TEST_INT='abc' is not a valid integer"):
        _get_int("TEST_INT", 0.5)


def test_get_int_rejects_invalid_value_float(monkeypatch):
    monkeypatch.setenv("TEST_INT", "0.67")
    with pytest.raises(ValueError, match="TEST_INT='0.67' is not a valid integer"):
        _get_int("TEST_INT", 0.67)
