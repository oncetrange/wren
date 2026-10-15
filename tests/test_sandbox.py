"""Shell commands in an OS sandbox: what they may write, read and reach."""

import os
import socket
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.config import ConfigError, PermissionRules, load_config
from wren.sandbox import CACHE_ENV, Sandbox, SandboxConfig, _unavailable
from wren.tools import ToolContext, default_tools
from wren.tools.shell import Bash, SandboxedBash

BASH = SandboxedBash()


def make(tmp_path: Path, platform: str = sys.platform, **config) -> Sandbox:
    project = tmp_path / "proj"
    (project / ".git" / "hooks").mkdir(parents=True, exist_ok=True)
    (project / ".git" / "config").write_text("[core]\n")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "id_ed25519").write_text("SECRET\n")
    return Sandbox(project, SandboxConfig(enabled=True, **config), secret_env=("MY_MODEL_KEY",),
                   home=home, platform=platform)


# --- what the sandbox is made of --------------------------------------------------


def test_seatbelt_profile(tmp_path):
    sb = make(tmp_path, "darwin")
    profile = sb.seatbelt_profile()
    assert profile.index("(deny file-write*)") < profile.index(f'(subpath "{sb.cwd}")')
    assert f'(subpath "{sb.cwd}/.git/hooks")' in profile
    assert '(remote ip "localhost:*")' in profile
    assert "(deny network*)" not in make(tmp_path, "darwin", network="allow").seatbelt_profile()
    assert sb.wrap(["bash", "-c", "ls"], sb.cwd)[:2] == ["sandbox-exec", "-p"]


def test_seatbelt_quotes_paths(tmp_path):
    odd = tmp_path / 'a "b" \\c'
    sb = Sandbox(odd, SandboxConfig(enabled=True), home=tmp_path, platform="darwin")
    assert '(subpath "' + str(sb.cwd).replace("\\", "\\\\").replace('"', '\\"') + '")' in sb.seatbelt_profile()


def test_bwrap_arguments(tmp_path):
    sb = make(tmp_path, "linux")
    args = sb.wrap(["bash", "-c", "ls"], sb.cwd / "sub")
    joined = " ".join(args)
    assert args[:5] == ["bwrap", "--ro-bind", "/", "/", "--dev"]
    assert f"--bind {sb.cwd} {sb.cwd}" in joined
    assert f"--ro-bind {sb.cwd}/.git/hooks {sb.cwd}/.git/hooks" in joined
    assert f"--tmpfs {sb.home}/.ssh" in joined
    assert "--unshare-net" in args and args[-4:] == ["--", "bash", "-c", "ls"]
    assert args[args.index("--chdir") + 1] == str(sb.cwd / "sub")
    assert "--unshare-net" not in make(tmp_path, "linux", network="allow").wrap(["true"], sb.cwd)


def test_container_engines_are_out_of_reach(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///elsewhere/docker.sock")
    (tmp_path / "home" / ".colima").mkdir(parents=True)
    mac, linux = make(tmp_path, "darwin"), make(tmp_path, "linux")
    deny = mac.seatbelt_profile().splitlines()[-1]
    assert deny.startswith("(deny network-outbound (require-any (remote unix-socket")
    assert "(require-not (require-any (remote unix-socket" in mac.seatbelt_profile()
    for path in (f"{mac.home}/.docker/run", "/elsewhere/docker.sock"):
        assert f'(remote unix-socket (subpath "{path}"))' in deny
    assert f"--tmpfs {linux.home}/.colima" in " ".join(linux.wrap(["true"], linux.cwd))
    assert "Docker" in mac.describe() and mac.blocked_hint("Cannot connect to the Docker daemon at unix://")


def test_secrets_leave_the_environment(tmp_path):
    env = make(tmp_path).env({"PATH": "/bin", "MY_MODEL_KEY": "k", "OPENAI_API_KEY": "k",
                              "GITHUB_TOKEN": "t", "DB_PASSWORD": "p", "HOME": "/h"})
    assert {k: v for k, v in env.items() if k not in CACHE_ENV} == {"PATH": "/bin", "HOME": "/h"}


def test_package_managers_get_caches_of_their_own(tmp_path):
    sb = make(tmp_path, allow_write=[str(tmp_path / "shared")])
    env = sb.env({"UV_CACHE_DIR": str(tmp_path / "shared" / "uv"), "PIP_CACHE_DIR": "/opt/pip"})
    assert env["UV_CACHE_DIR"] == str(tmp_path / "shared" / "uv")      # chosen, and writable: kept
    assert env["PIP_CACHE_DIR"] == str(sb.cache / "pip")               # not writable: replaced
    assert env["npm_config_cache"] == str(sb.cache / "npm")
    assert sb.cache.is_dir() and sb.cache in sb.writable(read_only=True)
    assert str(sb.cache) in sb.describe()


def test_hints_only_for_failures_that_look_like_the_sandbox(tmp_path):
    sb = make(tmp_path)
    assert "sandbox: false" in sb.blocked_hint("touch: /x: Operation not permitted")
    assert "no network" in sb.blocked_hint("curl: (6) Could not resolve host: pypi.org")
    assert sb.blocked_hint("AssertionError: 1 != 2") is None


def test_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[sandbox]\nenabled = true\nallow_write = ["~/.cache/uv"]\n')
    assert load_config(path).sandbox == SandboxConfig(enabled=True, allow_write=["~/.cache/uv"])
    for bad in ('network = "some"', "allow_write = '~/x'", "enabld = true", 'enabled = "yes"'):
        path.write_text(f"[sandbox]\n{bad}\n")
        with pytest.raises(ConfigError):
            load_config(path)


def test_the_way_out_only_exists_with_a_sandbox():
    assert "sandbox" not in Bash.input_schema["properties"] and "sandbox" not in Bash.description
    assert "sandbox" in SandboxedBash.input_schema["properties"]
    assert type(default_tools()[3]) is Bash and type(default_tools(sandboxed=True)[3]) is SandboxedBash


# --- leaving the sandbox needs approval ---------------------------------------------

OUT = {"command": "uv sync", "sandbox": False}
IN = {"command": "uv sync"}


def test_leaving_asks_in_every_mode_and_is_remembered_on_its_own(tmp_path):
    sb = make(tmp_path)
    for mode in ("ask", "accept_edits", "auto", "plan"):
        assert Permissions(mode=mode, sandbox=sb).needs_approval(BASH, OUT), mode
    p = Permissions(mode="plan", sandbox=sb)
    p.remember(BASH, OUT)
    assert p.needs_approval(BASH, OUT)                        # plan mode ignores "always"
    p = Permissions(mode="ask", sandbox=sb)
    assert not p.needs_approval(BASH, IN) and p.needs_approval(BASH, OUT)
    p.remember(BASH, OUT)
    assert p.key(BASH, OUT) == "unsandboxed:uv" and not p.needs_approval(BASH, {**OUT, "command": "uv lock"})
    # Without a sandbox, sandbox: false means nothing.
    assert Permissions(mode="auto").key(BASH, OUT) == "bash:uv"
    assert not Permissions(mode="auto").needs_approval(BASH, OUT)


def test_rules_for_leaving(tmp_path):
    rules = PermissionRules(allow=["bash:uv", "unsandboxed:git fetch"], deny=["bash:rm"])
    p = Permissions(mode="ask", sandbox=make(tmp_path), rules=rules)
    assert p.needs_approval(BASH, OUT)                        # bash:uv only covers it inside
    assert not p.needs_approval(BASH, {"command": "git fetch origin", "sandbox": False})
    assert p.blocked(BASH, {"command": "rm -rf x", "sandbox": False})


# --- for real ---------------------------------------------------------------------

needs_sandbox = pytest.mark.skipif(_unavailable(sys.platform) is not None,
                                   reason=f"no sandbox here: {_unavailable(sys.platform)}")


def run(sb: Sandbox, command: str, **args) -> str:
    out = BASH.run({"command": command, **args}, ToolContext(cwd=sb.cwd, sandbox=sb))
    return out.content


@needs_sandbox
def test_writes_stay_in_the_project_and_temp(tmp_path):
    sb = make(tmp_path)
    probe = Path.home() / f".wren-sandbox-probe-{uuid.uuid4().hex}"
    try:
        out = run(sb, f"echo hi > made.txt && echo tmp > \"$TMPDIR/x\" && cat made.txt; touch {probe}")
        assert "hi" in out and (sb.cwd / "made.txt").exists()
        assert not probe.exists() and "sandbox: false" in out
    finally:
        probe.unlink(missing_ok=True)
    out = run(sb, "echo evil >> .git/hooks/pre-commit; echo x >> .git/config")
    assert not (sb.cwd / ".git" / "hooks" / "pre-commit").exists()
    assert (sb.cwd / ".git" / "config").read_text() == "[core]\n"


@needs_sandbox
def test_credentials_and_keys_are_out_of_reach(tmp_path, monkeypatch):
    sb = make(tmp_path)
    monkeypatch.setenv("MY_MODEL_KEY", "sk-123")
    out = run(sb, f"cat {sb.home}/.ssh/id_ed25519; echo key=${{MY_MODEL_KEY:-none}}")
    assert "SECRET" not in out and "key=none" in out


@needs_sandbox
def test_no_network_but_localhost(tmp_path):
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    threading.Thread(target=lambda: server.accept()[0].close(), daemon=True).start()
    sb = make(tmp_path)
    py = sys.executable
    out = run(sb, f"{py} -c \"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=3); "
                  "print('local ok')\"; "
                  f"{py} -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\"")
    server.close()
    if sys.platform == "darwin":   # on Linux, localhost is the command's own
        assert "local ok" in out
    assert "Error" in out and "sandbox" in out


@needs_sandbox
def test_no_container_engine_socket(tmp_path, monkeypatch):
    short = Path(tempfile.mkdtemp(dir="/tmp"))           # unix socket paths are short
    try:
        engine, other = socket.socket(socket.AF_UNIX), socket.socket(socket.AF_UNIX)
        engine.bind(str(short / "docker.sock"))
        other.bind(str(short / "other.sock"))
        for s in (engine, other):
            s.listen()
        monkeypatch.setenv("DOCKER_HOST", f"unix://{short}/docker.sock")
        sb = make(tmp_path)
        (sb.cwd / "connect.py").write_text(
            "import socket, sys\n"
            "for p in sys.argv[1:]:\n"
            "    try:\n"
            "        socket.socket(socket.AF_UNIX).connect(p)\n"
            "        print(p[-11:], 'ok')\n"
            "    except OSError as e:\n"
            "        print(p[-11:], 'no', e)\n")
        out = run(sb, f"{sys.executable} connect.py {short}/docker.sock {short}/other.sock")
        assert "docker.sock no" in out and "other.sock ok" in out, out
    finally:
        for name in ("docker.sock", "other.sock"):
            (short / name).unlink(missing_ok=True)
        os.rmdir(short)


@needs_sandbox
def test_background_jobs_are_sandboxed_too(tmp_path):
    sb = make(tmp_path)
    ctx = ToolContext(cwd=sb.cwd, sandbox=sb)
    BASH.run({"command": "echo x >> .git/config", "run_in_background": True}, ctx)
    ctx.jobs.kill_all()
    assert (sb.cwd / ".git" / "config").read_text() == "[core]\n"


@needs_sandbox
def test_the_agent_leaves_the_sandbox_only_with_approval(tmp_path, model):
    sb = make(tmp_path)
    ctx = ToolContext(cwd=sb.cwd, sandbox=sb)
    provider = ScriptedProvider([
        call("bash", "t1", command="echo hook > .git/hooks/post-commit", sandbox=False),
        reply("done"),
    ])
    ui = RecordingUI()
    Agent(provider, model, ctx, ui, Permissions(mode="auto", sandbox=sb)).run("go")
    assert [e[1] for e in ui.events if e[0] == "confirm"] == ["bash"]   # asked, even in auto
    assert (sb.cwd / ".git" / "hooks" / "post-commit").read_text() == "hook\n"
    agent = Agent(ScriptedProvider([]), model, ctx, ui)
    assert "# Sandbox" in agent.system and "no network" in agent.system


def test_wren_home_is_followed(tmp_path, monkeypatch):
    monkeypatch.setenv("WREN_HOME", str(tmp_path / "wren"))
    sb = Sandbox(tmp_path, SandboxConfig(enabled=True))
    assert sb.cache == (tmp_path / "wren" / "sandbox-cache").resolve() and sb.cache.is_dir()
    assert (tmp_path / "wren" / "env").resolve() in sb.hidden
