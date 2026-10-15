"""The sandbox's proxy: only allowed hosts get through, and Linux sandboxes reach it via the bridge."""

import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import RecordingUI, ScriptedProvider, call, reply
from test_sandbox import make, needs_sandbox

from wren.agent.loop import Agent
from wren.agent.permissions import Decision, Permissions, load_project_rules
from wren.config import ConfigError, PermissionRules, load_config
from wren.netproxy import PACKAGE_HOSTS, Proxy, expand, host_allowed
from wren.tools import ToolContext
from wren.tools.shell import SandboxedBash


@pytest.fixture
def upstream():
    """A local web server recording what it was asked."""
    seen: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            body = b"hello from upstream"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a): pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], seen
    server.shutdown()


def ask(proxy_port: int, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=5) as s:
        s.sendall(request)
        data = b""
        while chunk := s.recv(65536):
            data += chunk
        return data


def test_host_patterns():
    allowed = expand(["@packages", "Example.com", "*.corp.dev"])
    assert set(PACKAGE_HOSTS) <= set(allowed)
    assert host_allowed("pypi.org", allowed) and host_allowed("EXAMPLE.com.", allowed)
    assert host_allowed("git.corp.dev", allowed) and not host_allowed("corp.dev", allowed)
    assert not host_allowed("pypi.org.evil.com", allowed) and not host_allowed("evilexample.com", allowed)


def test_config_takes_a_host_list(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[sandbox]\nnetwork = ["@packages", "*.github.com"]\n')
    assert load_config(path).sandbox.network == ["@packages", "*.github.com"]
    for bad in ('["https://pypi.org"]', '["pypi.org/simple"]', '[1]', '"some"'):
        path.write_text(f"[sandbox]\nnetwork = {bad}\n")
        with pytest.raises(ConfigError):
            load_config(path)


def test_plain_http_is_forwarded_one_request_per_connection(upstream):
    port, seen = upstream
    proxy = Proxy(["127.0.0.1"])
    try:
        reply = ask(proxy.port, f"GET http://127.0.0.1:{port}/a?b=1 HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                "Proxy-Connection: keep-alive\r\nX-Kept: yes\r\n\r\n".encode())
    finally:
        proxy.close()
    assert reply.startswith(b"HTTP/1.0 200") and reply.endswith(b"hello from upstream")
    path, headers = seen[0]
    assert path == "/a?b=1" and headers["X-Kept"] == "yes" and headers["Connection"] == "close"
    assert "Proxy-Connection" not in headers


def test_connect_tunnels_to_allowed_hosts_only(upstream):
    port, _ = upstream
    proxy = Proxy(["127.0.0.1"])
    try:
        reply = ask(proxy.port, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n"
                                "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
        assert reply.startswith(b"HTTP/1.1 200 Connection established") and b"hello from upstream" in reply
        refused = ask(proxy.port, b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
        assert refused.startswith(b"HTTP/1.1 403") and b"evil.example" in refused
        assert ask(proxy.port, b"GET http://evil.example/ HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 403")
        assert proxy.refused_since(0) == ["evil.example"]
    finally:
        proxy.close()


def test_the_bridge_reaches_the_proxy_through_its_socket(upstream):
    port, seen = upstream
    proxy = Proxy(["127.0.0.1"], unix_socket=True)
    free = socket.create_server(("127.0.0.1", 0))
    bridge_port = free.getsockname()[1]
    free.close()
    fetch = ("import urllib.request as u, time; time.sleep(0.3); "
             f"o = u.build_opener(u.ProxyHandler({{'http': 'http://127.0.0.1:{bridge_port}'}})); "
             f"print(o.open('http://127.0.0.1:{port}/via-bridge').read().decode())")
    try:
        out = subprocess.run([sys.executable, "-m", "wren.netbridge", str(proxy.unix_path), str(bridge_port),
                              "--", sys.executable, "-c", fetch], capture_output=True, text=True, timeout=30)
    finally:
        proxy.close()
    assert out.returncode == 0, out.stderr
    assert "hello from upstream" in out.stdout and seen[0][0] == "/via-bridge"


@needs_sandbox
def test_sandboxed_commands_get_the_allowed_hosts_and_nothing_else(tmp_path, upstream):
    port, seen = upstream
    sb = make(tmp_path, network=["127.0.0.1"])
    ctx = ToolContext(cwd=sb.cwd, sandbox=sb)
    py = sys.executable
    # Through the proxy, though the upstream is on 127.0.0.1, which NO_PROXY would send direct.
    # On Linux going direct can't work (the command has a network namespace of its own), so
    # reaching the upstream there proves the way through the bridge.
    fetch = ("import os, urllib.request as u; [os.environ.pop(k, None) for k in ('NO_PROXY', 'no_proxy')]; "
             "o = u.build_opener(u.ProxyHandler({{'http': os.environ['HTTP_PROXY']}})); "
             "print(o.open('{url}').read().decode())")
    ok = SandboxedBash().run({"command": f"{py} -c \"{fetch.format(url=f'http://127.0.0.1:{port}/in')}\""}, ctx)
    assert "hello from upstream" in ok.content and seen[-1][0] == "/in", ok.content
    no = SandboxedBash().run({"command": f"{py} -c \"{fetch.format(url='http://blocked.invalid/')}\""}, ctx)
    assert no.is_error and "The proxy refused blocked.invalid" in no.content, no.content
    assert "network only to 127.0.0.1" in sb.describe()


# --- asking about hosts ---------------------------------------------------------


def test_the_users_say_beats_the_allowlist():
    say = {"no.example": False, "yes.example": True}
    proxy = Proxy(["no.example", "@ask"], decide=say.get)
    try:
        assert proxy.allowed == ["no.example"]
        assert not proxy.permits("no.example") and proxy.permits("yes.example")
        assert not proxy.permits("other.example")
    finally:
        proxy.close()


def test_host_decisions_from_rules_and_answers(tmp_path):
    rules = PermissionRules(allow=["net:*.example.com", "bash:ls"], deny=["net:evil.example.com"])
    p = Permissions(rules=rules, sandbox=make(tmp_path, network="ask"), project_file=tmp_path / "rules.toml")
    assert p.sandbox is not None and p.sandbox.decide("a.example.com") is True   # bound to the sandbox
    assert p.host_decision("evil.example.com") is False                          # deny beats allow
    assert p.host_decision("github.com") is None
    p.answer_host("github.com", True, persist=True)
    p.answer_host("gitlab.com", False)
    assert p.host_decision("github.com") is True and p.host_decision("gitlab.com") is False
    assert load_project_rules(tmp_path / "rules.toml").allow == ["net:github.com"]


def test_ask_in_the_config(tmp_path):
    path = tmp_path / "config.toml"
    for network, asks, hosts in (('"ask"', True, []), ('["@packages", "@ask"]', True, ["@packages"]),
                                 ('["pypi.org"]', False, ["pypi.org"]), ('"deny"', False, [])):
        path.write_text(f"[sandbox]\nnetwork = {network}\n")
        sb = make(tmp_path, network=load_config(path).sandbox.network)
        assert (sb.asks, sb.allowed_hosts, sb.proxied) == (asks, hosts, asks or bool(hosts))
    assert "the user is asked" in make(tmp_path, network="ask").describe()


class HostUI(RecordingUI):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def allow_host(self, host):
        self.events.append(("host", host))
        return self.answers.pop(0)


@needs_sandbox
def test_a_refused_host_is_asked_about_once_the_command_ends(tmp_path, upstream, model):
    port, seen = upstream
    sb = make(tmp_path, network="ask")
    fetch = ("import os, urllib.request as u; [os.environ.pop(k, None) for k in ('NO_PROXY', 'no_proxy')]; "
             "o = u.build_opener(u.ProxyHandler({'http': os.environ['HTTP_PROXY']})); "
             f"print(o.open('http://127.0.0.1:{port}/in').read().decode())")
    command = f"{sys.executable} -c \"{fetch}\""
    provider = ScriptedProvider([call("bash", "t1", command=command), call("bash", "t2", command=command),
                                 reply("done")])
    ui = HostUI([Decision(allow=True)])
    Agent(provider, model, ToolContext(cwd=sb.cwd, sandbox=sb), ui,
          Permissions(mode="ask", sandbox=sb)).run("fetch it")
    assert [e for e in ui.events if e[0] == "host"] == [("host", "127.0.0.1")]   # asked once
    first, second = (e[2] for e in ui.events if e[0] == "result")
    assert first.is_error and "The user allowed 127.0.0.1" in first.content, first.content
    assert "sandbox: false" not in first.content                       # asked instead
    assert "hello from upstream" in second.content and seen[-1][0] == "/in", second.content
