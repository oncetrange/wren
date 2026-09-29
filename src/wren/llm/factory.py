from wren.config import ModelConfig
from wren.llm.base import Provider


def create_provider(cfg: ModelConfig) -> Provider:
    if cfg.provider == "anthropic":
        from wren.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.provider == "openai":
        from wren.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"unknown provider {cfg.provider!r} for model {cfg.name!r}")
