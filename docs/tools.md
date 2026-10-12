# Background commands and the web

## Background commands

For commands that keep running (a dev server, a watcher, a long build) the model runs `bash` with
`run_in_background`: it gets a job id right away, reads new output with `bash_output` (optionally
waiting for the job to finish or for a line like "listening on" to appear) and stops the job with
`kill_job`. It is told when a job ends. Jobs run in their own process group and are stopped when
wren exits (a subagent's when the subagent finishes); `/jobs` lists them, shows their output or
stops one, and the bottom line counts the running ones.

## Web access

`web_search` searches the web and `web_fetch` reads a page as text (HTML reduced to headings,
lists, tables, code and links; long pages in parts). Both ask for approval, `web_fetch` per
domain, since a URL can carry out anything the model has read. Searching uses Brave or Tavily
when `$BRAVE_API_KEY` or `$TAVILY_API_KEY` is set, otherwise DuckDuckGo's HTML page (no key, but
best-effort). To choose, or to turn web access off:

```toml
[web]
search = "tavily"                  # brave, tavily or duckduckgo
api_key_env = "MY_TAVILY_KEY"      # default: BRAVE_API_KEY / TAVILY_API_KEY
# enabled = false                  # no web tools (or run with --no-web)
```

The `general` subagent gets the web tools too; `explore` doesn't. Benchmark runs through the Pier
adapter have no web access: the tasks come from public repositories whose real fixes are online.
