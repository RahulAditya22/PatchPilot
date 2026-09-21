from __future__ import annotations

import pytest

from patchpilot.config import Config, get_config


def test_config_defaults() -> None:
    """Verify default values in Config."""
    config = Config()
    assert config.MAX_RETRIES == 3
    assert config.SANDBOX_MODE == "subprocess"


def test_get_config_returns_same_instance() -> None:
    """Call get_config twice, verify same object."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set env vars, verify config picks them up."""
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    config = Config()
    assert config.MAX_RETRIES == 5
    assert config.SANDBOX_MODE == "docker"
