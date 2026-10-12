# Models

| name | model | endpoint | key |
|---|---|---|---|
| `qwen` | qwen3-coder-plus | DashScope, Anthropic-compatible | `$DASHSCOPE_API_KEY` |
| `kimi` (default) | kimi-k2.7-code (thinking always on) | Moonshot, Anthropic-compatible | `$MOONSHOT_API_KEY` |
| `claude` | claude-opus-5 | Anthropic | `$ANTHROPIC_API_KEY` |
| `qwen-openai` | qwen3-coder-plus | DashScope, OpenAI-compatible | `$DASHSCOPE_API_KEY` |
| `deepseek` | deepseek-chat (no price set) | DeepSeek, OpenAI-compatible | `$DEEPSEEK_API_KEY` |

API keys come from the environment, or from `~/.wren/env` (`KEY=value` lines, readable only by
you; `wren setup` writes it). A variable set in your shell wins over the file. When the default
model has no key but another model does, wren uses that one and says so.

Override builtins or add any Anthropic- or OpenAI-compatible model in `~/.wren/config.toml`:

```toml
default_model = "kimi"

[models.kimi]                      # fields merge over the builtin
base_url = "https://api.moonshot.ai/anthropic"   # international platform key

[models.local]                     # any Anthropic-compatible endpoint
model = "my-model"
base_url = "http://localhost:8000"
api_key_env = "LOCAL_API_KEY"
auth = "bearer"                    # send the key as Authorization: Bearer
prompt_cache = false
thinking = { type = "enabled", budget_tokens = 8000 }
price = { input = 0.6, output = 2.5 }   # USD per million tokens, enables cost display

[models.ollama]                    # any OpenAI-compatible chat completions endpoint
provider = "openai"
model = "qwen3-coder:30b"
base_url = "http://localhost:11434/v1"
api_key_env = ""                   # no key needed (default: $OPENAI_API_KEY)
context_window = 65536
reasoning_effort = "medium"        # optional, sent as is
extra_body = { enable_thinking = true }   # optional vendor-specific request fields
replay_reasoning = true            # send reasoning back with tool calls (DeepSeek/Kimi thinking)
```

With `provider = "openai"`, wren speaks Chat Completions: tool results become `tool` messages,
tool errors are marked with `Error:`, reasoning streams as thinking, and cached prompt tokens are
read from the usage report (caching itself is automatic). For OpenAI's own endpoint the output
limit goes in `max_completion_tokens`; set `max_tokens_param` to override.

Models are only read from the user config, never from the project directory, so a repository can't redirect your API key to its own endpoint.
