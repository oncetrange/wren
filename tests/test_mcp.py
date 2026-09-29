"""MCP: server config, connecting over stdio, and MCP tools in the agent."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RecordingUI, ScriptedProvider, reply
from fake_anthropic import FakeAnthropic, text_turn, tool_turn

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions
from wren.agent.subagents import builtin_agent_types
from wren.config import ConfigError, McpServerConfig, ModelConfig, load_config, parse_mcp_servers
from wren.llm.types import Message, Response, ToolResultBlock, ToolUseBlock, Usage
from wren.mcp_servers import McpServers, load_project_mcp
from wren.tools.mcp import tool_name

SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


def fake_server(name="fake", **kw) -> McpServerConfig:
    return McpServerConfig(name, command=sys.executable, args=[SERVER], **kw)


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    m = McpServers([fake_server(timeout=3), McpServerConfig("broken", command="/nonexistent/server")],
                   Path.cwd(), log_dir=tmp_path_factory.mktemp("logs"))
    m.connect(timeout=20)
    yield m
    m.close()


def use(id, name, **input):
    return Response(Message("assistant", [ToolUseBlock(id, name, input)]), "tool_use", Usage(10, 5))


def results(agent):
    return {b.tool_use_id: b for m in agent.messages for b in m.content if isinstance(b, ToolResultBlock)}


# --- config ------------------------------------------------------------------------

def test_parse_servers(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.delenv("MISSING", raising=False)
    servers = parse_mcp_servers({
        "files": {"command": "npx", "args": ["-y", "server-files", "${MISSING:-/tmp}"],
                  "env": {"TOKEN": "${GH_TOKEN}"}},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${GH_TOKEN}"},
                   "timeout": 30},
        "off": {"command": "x", "enabled": False},
    }, source="user")
    assert [s.name for s in servers] == ["files", "remote"]
    files, remote = servers
    assert files.transport == "stdio" and files.args == ["-y", "server-files", "/tmp"]
    assert files.env == {"TOKEN": "secret"}
    assert remote.transport == "http" and remote.headers == {"Authorization": "Bearer secret"}
    assert remote.timeout == 30


@pytest.mark.parametrize("spec, error", [
    ({"type": "sse", "url": "http://x"}, "SSE"), ({"args": ["x"]}, "needs a 'command'"),
    ({"type": "http"}, "needs a 'url'"), ({"command": "x", "timeout": 0}, "timeout"),
    ({"command": "x", "cwd": "/"}, "unknown keys"), ({"type": "ws", "url": "x"}, "type must be"),
])
def test_invalid_servers(spec, error):
    with pytest.raises(ConfigError, match=error):
        parse_mcp_servers({"s": spec}, source="user")
    with pytest.raises(ConfigError, match="names may only"):
        parse_mcp_servers({"a b": {"command": "x"}}, source="user")


def test_user_config_and_project_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[mcp.db]\ncommand = "db-server"\nargs = ["--ro"]\n')
    assert load_config(path).mcp[0].args == ["--ro"]
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"docs": {"type": "http", "url": "https://docs.example/mcp"}}}))
    project = load_project_mcp(tmp_path)
    assert project.servers[0].url == "https://docs.example/mcp"
    assert project.servers[0].source == ".mcp.json" and len(project.digest) == 64
    (tmp_path / ".mcp.json").write_text("{not json")
    with pytest.raises(ConfigError):
        load_project_mcp(tmp_path)


def test_tool_names():
    assert tool_name("git hub", "list.issues") == "mcp__git_hub__list_issues"
    assert len(tool_name("s", "x" * 100)) == 64


# --- servers -----------------------------------------------------------------------

def test_connects_and_lists_tools(servers):
    fake, broken = servers.servers["fake"], servers.servers["broken"]
    assert fake.status == "connected"
    assert {t.name: t.read_only for t in fake.tools} == {"echo": False, "add": True, "fail": False,
                                                          "slow": False}
    assert "Use echo to repeat text back." in servers.instructions()
    assert broken.status == "failed" and "FileNotFoundError" in broken.error
    assert "fake server starting" in (servers.log_dir / "mcp-fake.log").read_text()


def test_agent_uses_mcp_tools(ctx, servers):
    turns = [use("c1", "mcp__fake__add", a=2, b=3), use("c2", "mcp__fake__echo", text="hi"),
             use("c3", "mcp__fake__fail"), use("c4", "mcp__fake__slow", seconds=10),
             use("c5", "mcp__broken__x"), reply("done")]
    agent = Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"), ctx,
                  RecordingUI([Decision(allow=True)] * 3), Permissions(mode="ask"), mcp=servers)
    assert "mcp__fake__echo" in agent.tools and "Use echo to repeat" in agent.system
    agent.run("go")
    r = results(agent)
    assert r["c1"].content == "5" and not r["c1"].is_error
    assert r["c2"].content == "echo: hi"
    assert r["c3"].is_error
    assert r["c4"].is_error and "timed out after 3s" in r["c4"].content
    assert r["c5"].is_error and "unknown tool" in r["c5"].content
    # add is read-only (no prompt); echo, fail and slow asked.
    assert [e[1] for e in agent.ui.events if e[0] == "confirm"] == [
        "mcp__fake__echo", "mcp__fake__fail", "mcp__fake__slow"]


def test_only_writing_subagents_get_mcp_tools(ctx, servers):
    class ToolRecording(ScriptedProvider):
        def stream(self, *, system, messages, tools):
            self.seen = [*getattr(self, "seen", []), {t.name for t in tools}]
            yield from super().stream(system=system, messages=messages, tools=tools)

    def task(kind, id):
        return use(id, "task", description="d", prompt=f"p {kind}", agent=kind)

    agent = Agent(ToolRecording([task("general", "t1"), reply("r"), task("explore", "t2"), reply("r"),
                                 reply("done")]),
                  ModelConfig(name="fake", model="f"), ctx, RecordingUI(), Permissions(mode="auto"),
                  mcp=servers, agent_types=builtin_agent_types())
    agent.run("go")
    general, explore = agent.provider.seen[1], agent.provider.seen[3]
    assert "mcp__fake__add" in general and not any(n.startswith("mcp__") for n in explore)


# --- CLI ---------------------------------------------------------------------------

def test_project_servers_need_trust(tmp_path):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (project / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"fake": {"command": sys.executable, "args": [SERVER]}}}))

    def run(server, *flags):
        (home / "config.toml").write_text(f'[models.fake]\nmodel = "m"\nbase_url = "{server.url}"\n'
                                          'api_key_env = "K"\nprompt_cache = false\n')
        env = {**os.environ, "WREN_HOME": str(home), "K": "k"}
        return subprocess.run([sys.executable, "-m", "wren.cli.main", "-p", "add", "-m", "fake", "--yolo",
                               "--no-checkpoints", "--no-final-check", *flags],
                              cwd=project, env=env, capture_output=True, text=True, timeout=60)

    with FakeAnthropic([text_turn("no tools")]) as server:
        proc = run(server)
    assert proc.returncode == 0, proc.stderr
    assert "skipping untrusted MCP servers" in proc.stdout
    assert not any(t["name"].startswith("mcp__") for t in server.requests[0]["body"]["tools"])

    with FakeAnthropic([tool_turn("mcp__fake__add", {"a": 1, "b": 2}), text_turn("3")]) as server:
        proc = run(server, "--trust-project-mcp")
    assert proc.returncode == 0, proc.stderr
    result = server.requests[1]["body"]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["content"] == "3"
    # Trusted now: the next run starts them without the flag.
    with FakeAnthropic([text_turn("ok")]) as server:
        run(server)
    assert any(t["name"] == "mcp__fake__add" for t in server.requests[0]["body"]["tools"])


def test_streamable_http(tmp_path):
    import socket
    import time

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, SERVER, str(port)], stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):  # wait for it to listen
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        m = McpServers([McpServerConfig("web", url=f"http://127.0.0.1:{port}/mcp",
                                        headers={"X-Test": "1"})], tmp_path, log_dir=tmp_path)
        m.connect(timeout=20)
        state = m.servers["web"]
        assert state.status == "connected", state.error
        assert "add" in {t.name for t in state.tools}
        assert m.call("web", "add", {"a": 4, "b": 5}).content[0].text == "9"
        m.close()
    finally:
        proc.terminate()
        proc.wait(5)
