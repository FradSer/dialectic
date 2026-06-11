"""Unit tests for the model-config factory (no network)."""

from google.adk.models.lite_llm import LiteLlm

from dialectica.llm_config import _DEFAULT_MODEL, get_model_config


def test_google_provider_returns_bare_model_name(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "google:gemini-3.5-flash")
    assert get_model_config() == "gemini-3.5-flash"


def test_missing_config_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("DEFAULT_MODEL_CONFIG", raising=False)
    assert get_model_config() == _DEFAULT_MODEL


def test_malformed_config_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "no-colon-here")
    assert get_model_config() == _DEFAULT_MODEL


def test_unknown_provider_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "mystery:model-x")
    assert get_model_config() == _DEFAULT_MODEL


def test_role_override_beats_default(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "google:gemini-3.5-flash")
    monkeypatch.setenv("JUDGE_MODEL_CONFIG", "google:gemini-3.1-pro-preview")
    assert get_model_config("Judge") == "gemini-3.1-pro-preview"


def test_openai_provider_builds_litellm_when_credentialed(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "openai:qwen3.6-35b-a3b")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE", "http://example/v1")
    config = get_model_config()
    assert isinstance(config, LiteLlm)
    assert config.model == "openai/qwen3.6-35b-a3b"


def test_openai_provider_without_credentials_falls_back(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "openai:gpt-4o")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert get_model_config() == _DEFAULT_MODEL


def test_openrouter_without_key_falls_back(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_CONFIG", "openrouter:some/model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert get_model_config() == _DEFAULT_MODEL
