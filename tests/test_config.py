import pytest

from wren.config import ConfigError, load_config
from wren.llm.anthropic_provider import _thinking_param
from wren.llm.types import Usage


def test_builtins_without_config_file(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.default_model == "qwen"
    kimi = config.model("kimi")
    assert kimi.model == "kimi-k2.7-code" and kimi.auth == "bearer"
    assert _thinking_param(kimi.thinking) == {"type": "enabled", "budget_tokens": 16000}
    assert kimi.thinking["budget_tokens"] < kimi.max_tokens


def test_user_entries_merge_over_builtins(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'default_model = "kimi"\n'
        "[models.kimi]\n"
        'base_url = "https://api.moonshot.ai/anthropic"\n'
        "thinking = { type = \"enabled\", budget_tokens = 8000 }\n"
        "[models.local]\n"
        'model = "my-model"\n'
        'base_url = "http://localhost:8000"\n'
    )
    config = load_config(path)
    kimi = config.model()
    assert kimi.base_url == "https://api.moonshot.ai/anthropic"
    assert kimi.model == "kimi-k2.7-code"  # untouched fields keep the builtin value
    assert kimi.thinking == {"type": "enabled", "budget_tokens": 8000}
    assert config.model("local").price is None


def test_string_thinking_is_expanded():
    assert _thinking_param("adaptive") == {"type": "adaptive"}


def test_price_uses_cache_rates():
    kimi = load_config().models["kimi"]
    cost = kimi.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000))
    assert cost == pytest.approx(0.95 + 4.0 + 0.19)


def test_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[models.x]\nmodel = "m"\ntemprature = 1\n')
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)
