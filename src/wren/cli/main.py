"""The `wren` command: parse the command line, then run one headless prompt or
the interactive session. Subcommands: `wren setup`, `wren schedule`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console

from wren import __version__
from wren.cli import app
from wren.cli.commands import BUILTIN_NAMES
from wren.cli.repl import Repl
from wren.cli.ui import RichUI
from wren.config import ConfigError
from wren.credentials import load_env_file
from wren.settings import Settings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    load_env_file()  # keys saved by `wren setup`; the shell's own variables win
    if argv[:1] == ["setup"]:
        from wren.cli.setup_cmd import setup_main

        return setup_main(argv[1:])
    if argv[:1] == ["schedule"]:
        from wren.cli.schedule_cmd import schedule_main

        return schedule_main(argv[1:])
    args = parser().parse_args(argv)

    json_output = bool(args.prompt) and args.output_format == "json"
    if args.prompt == "-":
        args.prompt = sys.stdin.read()
    settings = Settings.load()
    # In JSON mode stdout carries only the result object; everything else goes to stderr.
    ui = RichUI(Console(highlight=False, stderr=json_output), background=app.background(settings))
    try:
        session = app.start(args, settings, ui, Path.cwd().resolve())
    except ConfigError as e:
        ui.error(str(e))
        return 1
    try:
        if args.prompt:
            start = time.monotonic()
            result = app.run_prompt(session.agent, args.prompt, BUILTIN_NAMES)
            if json_output:
                print(json.dumps(app.result_json(session.agent, result, time.monotonic() - start),
                                 ensure_ascii=False))
            return 0 if session.agent.status == "done" else 1
        return Repl(session.agent, ui, session.config, settings, session.memories).loop()
    finally:
        session.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wren", description="A coding agent for your terminal.",
                                epilog="wren setup: choose a model and save its API key · "
                                       "wren schedule --help: run prompts on a cron schedule")
    p.add_argument("-m", "--model", help="model name from the config (default: config default_model)")
    p.add_argument("-c", "--continue", dest="continue_", action="store_true",
                   help="continue the most recent session in this directory")
    p.add_argument("-r", "--resume", nargs="?", const="", metavar="ID",
                   help="resume a session by id, or pick one from a list")
    p.add_argument("--mode", choices=["ask", "accept_edits", "plan", "auto"], default="ask",
                   help="permission mode to start in (Shift+Tab switches in a session)")
    p.add_argument("--plan", action="store_true", help="start in plan mode (same as --mode plan)")
    p.add_argument("--yolo", action="store_true", help="run every tool without asking (same as --mode auto)")
    p.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None,
                   help="use long-term memory: the model reads and keeps memories across "
                        "sessions (default: on, off with -p)")
    p.add_argument("--no-web", action="store_true", help="don't offer web_search and web_fetch")
    p.add_argument("--no-checkpoints", action="store_true", help="don't snapshot the workspace before each prompt")
    p.add_argument("--version", action="version", version=f"wren {__version__}")

    headless = p.add_argument_group("headless runs")
    headless.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                          help="run a single request non-interactively and exit ('-' reads stdin)")
    headless.add_argument("--output-format", choices=["text", "json"], default="text",
                          help="with -p: 'json' prints one JSON result object to stdout and "
                               "sends progress output to stderr")
    headless.add_argument("--max-turns", type=int, default=100, metavar="N",
                          help="stop after N model calls per request (default 100)")
    headless.add_argument("--time-limit", type=float, metavar="MINUTES",
                          help="stop after this long, reminding the model to save its work as the limit "
                               "nears (set it a little under a harness's own timeout)")
    headless.add_argument("--final-check", action=argparse.BooleanOptionalAction, default=None,
                          help="before finishing a request that changed files, have the model re-check "
                               "the request's explicit instructions; also warns when --max-turns is "
                               "nearly used up (default: on with -p, off otherwise)")
    headless.add_argument("--trust-project-hooks", action="store_true",
                          help="run the project's .wren/hooks.toml without asking (needed with -p)")
    headless.add_argument("--trust-project-mcp", action="store_true",
                          help="start the project's .mcp.json servers without asking (needed with -p)")

    experiments = p.add_argument_group("experiments")
    experiments.add_argument("--mask-at", type=int, metavar="TOKENS",
                             help="clear old tool outputs past this prompt size (0: never); overrides "
                                  "the model's")
    experiments.add_argument("--compact-at", type=int, metavar="TOKENS",
                             help="summarize the conversation past this prompt size; overrides the model's")
    experiments.add_argument("--no-subagents", action="store_true", help="don't offer the task tool")
    return p


if __name__ == "__main__":
    sys.exit(main())
