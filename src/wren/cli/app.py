"""Assembling a session from the command line: the model, the agent and what
plugs into it (skills, subagents, memory, MCP servers, hooks, extra tools).

`start` is the one place an interactive or headless session is put together;
`run_prompt` is how both send a prompt.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from wren import mentions
from wren.agent import shell_hooks
from wren.agent.loop import Agent
from wren.agent.permissions import Mode, Permissions
from wren.agent.session import SESSIONS_DIR, SessionLog, SessionState, list_sessions, load_session
from wren.agent.subagents import discover_agent_types
from wren.checkpoint import Checkpoints
from wren.cli.pickers import pick
from wren.cli.setup_cmd import NoUsableModel, default_model, run_setup
from wren.cli.terminal import detect_background
from wren.cli.text import session_options
from wren.cli.ui import RichUI
from wren.config import Config, ConfigError, ModelConfig, load_config
from wren.llm.base import Provider
from wren.llm.factory import create_provider
from wren.mcp_servers import PROJECT_MCP, McpServers, load_project_mcp
from wren.memory import Memories
from wren.settings import Settings
from wren.skills import discover
from wren.skills import expand as expand_skill
from wren.tools import ToolContext
from wren.tools.schedule import ScheduleTool
from wren.tools.web import SearchConfig, WebFetch, WebSearch


@dataclass
class Session:
    agent: Agent
    config: Config
    memories: Memories
    mcp: McpServers | None

    def close(self) -> None:
        self.agent.end_session()
        if self.mcp:
            self.mcp.close()


def start(args: argparse.Namespace, settings: Settings, ui: RichUI, cwd: Path) -> Session:
    """Everything the command line asks for, ready to run. Raises ConfigError."""
    config = load_config()
    state = session_to_resume(args, cwd)
    config, model = choose_model(args, config, state, ui)
    provider = create_provider(model)
    if ignored := model.ignored_options():
        ui.notice(f"model {model.name!r}: {', '.join(ignored)} not used with the {model.provider} provider")

    skills, skill_warnings = discover(cwd)
    agent_types, agent_warnings = discover_agent_types(cwd)
    memories = Memories.for_project(cwd)
    for warning in skill_warnings + agent_warnings:
        ui.notice(warning)
    mcp = connect_mcp(config, cwd, args, ui)
    try:
        agent = Agent(
            provider,
            model,
            ToolContext(cwd=cwd),
            ui,
            permissions=Permissions(mode=permission_mode(args), allow_auto=permission_mode(args) == "auto"),
            log=SessionLog(path=state.path) if state else SessionLog(),
            checkpoints=None if args.no_checkpoints else Checkpoints(cwd),
            max_turns=args.max_turns,
            final_check=bool(args.prompt) if args.final_check is None else args.final_check,
            skills=skills,
            agent_types={} if args.no_subagents else agent_types,
            memory=memories if (not args.prompt if args.memory is None else args.memory) else None,
            mcp=mcp,
        )
        agent.auto_memory = settings.memory_auto
        if args.time_limit:
            agent.set_time_limit(args.time_limit)
        if not args.prompt:  # scheduled jobs are made in sessions, not by (scheduled) headless runs
            agent.tools["schedule"] = ScheduleTool()
        if config.web.enabled and not args.no_web:
            search = SearchConfig(config.web.search, config.web.api_key_env)
            agent.tools.update({"web_search": WebSearch(search), "web_fetch": WebFetch()})
        agent.resolve_model = lambda name: _provider_for(config.model(name))
        if state:
            agent.restore(state)
            ui.render_history(agent.messages, agent.tools, agent.ctx)
        hooks = config.hooks
        project_hooks = shell_hooks.load_project_hooks(cwd)
        if project_hooks and trust_project_hooks(project_hooks, args, ui):
            hooks = hooks + project_hooks.specs
        shell_hooks.install(agent, hooks)
    except BaseException:
        if mcp:
            mcp.close()
        raise
    agent.start_session("resume" if state else "startup")
    return Session(agent, config, memories, mcp)


def choose_model(args: argparse.Namespace, config: Config, state: SessionState | None,
                 ui: RichUI) -> tuple[Config, ModelConfig]:
    """The model to use: -m, the resumed session's, or the default (falling back
    to one with a key, or setting one up on a first interactive start)."""
    name = args.model
    if name is None and state and state.model in config.models:
        name = state.model
    if name is not None or os.environ.get("WREN_MODEL"):
        model = config.model(name)
    else:
        try:
            model, note = default_model(config)
        except NoUsableModel as e:
            if args.prompt or not ui.interactive:
                raise ConfigError(str(e)) from None
            ui.console.print("[bold]Welcome to wren.[/] First, a model to talk to.")
            chosen = run_setup(ui.console)
            if chosen is None:
                raise ConfigError("no model set up; run `wren setup` when you're ready") from None
            config = load_config()
            model, note = config.models[chosen], None
        if note:
            ui.notice(note)
    if args.mask_at is not None or args.compact_at is not None:  # for experiments
        model = dataclasses.replace(
            model, mask_at=model.mask_at if args.mask_at is None else args.mask_at,
            compact_at=model.compact_at if args.compact_at is None else args.compact_at)
    return config, model


def _provider_for(model: ModelConfig) -> tuple[Provider, ModelConfig]:
    return create_provider(model), model


def run_prompt(agent: Agent, text: str, reserved: AbstractSet[str] = frozenset()) -> str:
    """Run a prompt, expanding `/skill-name arguments` into the skill (unless the
    name is in `reserved`, the built-in commands) and attaching @-mentioned files."""
    attachments = []
    if invoked := expand_skill(text, agent.skills, reserved):
        skill, arguments = invoked
        agent.log.record("skill", name=skill.name, by="user")
        attachments.append(skill.invocation(arguments))
    if found := mentions.find(text, agent.ctx.cwd):
        files, shown = mentions.attach(found, agent.ctx)
        attachments += files
        show = getattr(agent.ui, "attached", None)
        for label, summary in shown:
            if show:
                show(label, summary)
        agent.log.record("mentions", paths=[label for label, _ in shown])
    return agent.run(text, attachments=attachments or None)


def result_json(agent: Agent, result: str, seconds: float) -> dict:
    """The headless result object (--output-format json)."""
    u = agent.usage
    return {
        "status": agent.status,
        "result": result,
        "model": agent.model.name,
        "model_id": agent.model.model,
        "session_id": agent.log.id,
        "session_log": str(agent.log.path) if agent.log.path else None,
        "turns": agent.turns,
        "tool_calls": agent.tool_calls,
        "tool_errors": agent.tool_errors,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_write_tokens": u.cache_write_tokens,
        },
        "cost_usd": agent.cost,
        "subagents": agent.subagent_runs,
        "todos": [t.to_dict() for t in agent.conv.todos],
        "plan": agent.plan_text,
        "plan_file": str(agent.plan_file) if agent.plan_file else None,
        "duration_s": round(seconds, 1),
    }


# --- project files that run things -------------------------------------------------


def connect_mcp(config: Config, cwd: Path, args: argparse.Namespace, ui: RichUI) -> McpServers | None:
    """Connect to the user's MCP servers and the project's trusted ones. Raises ConfigError."""
    servers = list(config.mcp)
    project = load_project_mcp(cwd)
    if project and project.servers and _trusted(
            project, granted=args.trust_project_mcp, args=args, ui=ui,
            skipping=f"skipping untrusted MCP servers in {PROJECT_MCP} (pass --trust-project-mcp to start them)",
            header=f"[bold]This project defines MCP servers[/] in {PROJECT_MCP}; "
                   "they run programs or connect to services:",
            lines=[f"  [cyan]{s.name}[/]: {escape(s.url or ' '.join([s.command or '', *s.args]))}"
                   for s in project.servers],
            question="Start these servers?"):
        taken = {s.name for s in servers}
        servers += [s for s in project.servers if s.name not in taken]
    if not servers:
        return None
    mcp = McpServers(servers, cwd)
    with ui.console.status(f"connecting to MCP server{'s' if len(servers) > 1 else ''}…"):
        mcp.connect()
    for state in mcp.servers.values():
        if state.status != "connected":
            ui.notice(f"MCP server {state.config.name!r} unavailable: {state.error} (see /mcp)")
    return mcp


def trust_project_hooks(project: shell_hooks.ProjectHooks, args: argparse.Namespace, ui: RichUI) -> bool:
    """Project hooks run only once their exact content is trusted."""
    return _trusted(
        project, granted=args.trust_project_hooks, args=args, ui=ui,
        skipping=f"skipping untrusted project hooks in {shell_hooks.PROJECT_HOOKS} "
                 "(pass --trust-project-hooks to run them)",
        header=f"[bold]This project defines hooks[/] in {shell_hooks.PROJECT_HOOKS}; "
               "they run shell commands on your machine:",
        lines=[f"  [cyan]{s.event}[/]" + (f" [dim]({s.matcher})[/]" if s.matcher else "")
               + f": {escape(s.command)}" for s in project.specs],
        question="Run these hooks?")


def _trusted(project: shell_hooks.Trustable, *, granted: bool, args: argparse.Namespace, ui: RichUI,
             skipping: str, header: str, lines: list[str], question: str) -> bool:
    """A project file that runs things is used only once its exact content is
    trusted: already, by the command-line flag, or by asking (it asks again
    whenever the file changes)."""
    if shell_hooks.is_trusted(project):
        return True
    if granted:
        shell_hooks.trust(project)
        return True
    if args.prompt or not ui.interactive:
        ui.notice(skipping)
        return False
    ui.console.print(header)
    for line in lines:
        ui.console.print(line)
    if pick(question, [(True, "Yes, trust this file (asks again if it changes)"),
                       (False, "No, skip them this time")], default=False):
        shell_hooks.trust(project)
        return True
    return False


# --- small decisions from the command line -----------------------------------------


def permission_mode(args: argparse.Namespace) -> Mode:
    return "auto" if args.yolo else "plan" if args.plan else args.mode


def background(settings: Settings) -> str:
    if settings.theme != "auto":
        return settings.theme
    return detect_background() or "dark"


def session_to_resume(args: argparse.Namespace, cwd: Path) -> SessionState | None:
    if args.resume:
        matches = list(SESSIONS_DIR.glob(f"*-{args.resume}.jsonl"))
        if not matches:
            raise ConfigError(f"no session with id {args.resume!r}")
        return load_session(matches[0])
    if not args.continue_ and args.resume is None:
        return None
    sessions = list_sessions(cwd)
    if not sessions:
        raise ConfigError(f"no previous sessions in {cwd}")
    if args.continue_:
        return sessions[0]
    state = pick("Resume which session?", session_options(sessions))
    if state is None:
        raise ConfigError("no session selected")
    return state
