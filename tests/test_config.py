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


@pytest.mark.parametrize("value", ["0", "-1.5"])
def test_get_float_positive_rejects_zero_and_negative(monkeypatch, value):
    monkeypatch.setenv("TEST_FLOAT", value)
    with pytest.raises(ValueError, match="must be greater than 0"):
        _get_float("TEST_FLOAT", 0.5, positive=True)


def test_get_float_positive_accepts_positive(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0.1")
    assert _get_float("TEST_FLOAT", 0.5, positive=True) == 0.1

#============================= INTEGER RETRIEVAL ====================================

def test_get_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert _get_int("TEST_INT", 0.5) == 0.5


def test_get_int_parses_env_value(monkeypatch):
    monkeypatch.setenv("TEST_INT", "67")
    result = _get_int("TEST_INT", 3)
    assert result == 67
    assert isinstance(result, int)


def test_get_int_rejects_whole_float(monkeypatch):
    monkeypatch.setenv("TEST_INT", "5.0")
    with pytest.raises(ValueError, match="TEST_INT='5.0' is not a valid integer"):
        _get_int("TEST_INT", 3)


def test_get_int_rejects_invalid_value_str(monkeypatch):
    monkeypatch.setenv("TEST_INT", "abc")
    with pytest.raises(ValueError, match="TEST_INT='abc' is not a valid integer"):
        _get_int("TEST_INT", 0.5)


def test_get_int_rejects_invalid_value_float(monkeypatch):
    monkeypatch.setenv("TEST_INT", "0.67")
    with pytest.raises(ValueError, match="TEST_INT='0.67' is not a valid integer"):
        _get_int("TEST_INT", 0.67)


def test_get_int_non_negative_rejects_negative(monkeypatch):
    monkeypatch.setenv("TEST_INT", "-1")
    with pytest.raises(ValueError, match="TEST_INT='-1' must be 0 or greater"):
        _get_int("TEST_INT", 3, non_negative=True)


def test_get_int_non_negative_accepts_zero(monkeypatch):
    monkeypatch.setenv("TEST_INT", "0")
    assert _get_int("TEST_INT", 3, non_negative=True) == 0
