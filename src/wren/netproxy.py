"""The sandbox's way to the network: an HTTP proxy that only lets through hosts
on the allowlist ([sandbox] network = ["pypi.org", ...]).

Sandboxed commands get HTTP(S)_PROXY pointing here and can't connect anywhere
else, so a host that isn't allowed is refused whichever program asks. HTTPS
goes through CONNECT and stays end-to-end encrypted: the proxy only sees the
host name. It listens on localhost, and on Linux also on a unix socket, which
netbridge.py exposes inside the sandbox's own network namespace.
"""

from __future__ import annotations

import os
import selectors
import socket
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

# Package registries, for `"@packages"` in the allowlist.
PACKAGE_HOSTS = (
    "pypi.org", "files.pythonhosted.org",                          # pip, uv
    "registry.npmjs.org", "registry.yarnpkg.com",                  # npm, yarn, pnpm
    "crates.io", "index.crates.io", "static.crates.io",            # cargo
    "proxy.golang.org", "sum.golang.org",                          # go
    "rubygems.org", "index.rubygems.org",                          # gem, bundler
    "repo.maven.apache.org", "repo1.maven.org",                    # maven, gradle
)
MAX_HEAD = 64 * 1024
IDLE_TIMEOUT = 300  # seconds a tunnel may sit silent


def expand(patterns: list[str]) -> list[str]:
    hosts: list[str] = []
    for p in patterns:
        hosts += PACKAGE_HOSTS if p == "@packages" else [p.lower()]
    return hosts


def host_allowed(host: str, patterns: list[str]) -> bool:
    """`example.com` is that host only; `*.example.com` its subdomains, not itself."""
    host = host.lower().rstrip(".")
    for p in patterns:
        if p.startswith("*."):
            if host.endswith(p[1:]):
                return True
        elif host == p:
            return True
    return False


class Proxy:
    def __init__(self, allowed: list[str], unix_socket: bool = False):
        self.allowed = expand(allowed)
        # Hosts refused, with when (time.monotonic()), most recent last.
        self.refused: deque[tuple[float, str]] = deque(maxlen=50)
        self._tcp = socket.create_server(("127.0.0.1", 0))
        self.port: int = self._tcp.getsockname()[1]
        self.unix_path: Path | None = None
        self._unix: socket.socket | None = None
        if unix_socket:
            self.unix_path = Path(tempfile.mkdtemp(prefix="wren-proxy-")) / "proxy.sock"
            self._unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._unix.bind(str(self.unix_path))
            self._unix.listen(64)
        self._closed = False
        for server in filter(None, (self._tcp, self._unix)):
            threading.Thread(target=self._accept, args=(server,), daemon=True, name="wren-proxy").start()

    def refused_since(self, since: float) -> list[str]:
        return list(dict.fromkeys(host for t, host in list(self.refused) if t >= since))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> dict[str, str]:
        """For the commands that should use it (most programs read these)."""
        no_proxy = "localhost,127.0.0.1,::1"
        return {"HTTP_PROXY": self.url, "HTTPS_PROXY": self.url, "http_proxy": self.url,
                "https_proxy": self.url, "NO_PROXY": no_proxy, "no_proxy": no_proxy}

    def close(self) -> None:
        self._closed = True
        for server in filter(None, (self._tcp, self._unix)):
            server.close()
        if self.unix_path is not None:
            self.unix_path.unlink(missing_ok=True)
            os.rmdir(self.unix_path.parent)

    def _accept(self, server: socket.socket) -> None:
        while not self._closed:
            try:
                client, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        upstream = None
        try:
            client.settimeout(30)
            head, rest = _read_head(client)
            if head is None:
                return
            request_line, _, headers = head.partition(b"\r\n")
            parts = request_line.split(b" ")
            if len(parts) != 3:
                return _reply(client, 400, "malformed request")
            method, target, version = parts
            if method == b"CONNECT":
                host, port = _split_host_port(target.decode("latin-1"), 443)
            else:
                url = urlsplit(target.decode("latin-1"))
                if url.scheme != "http" or not url.hostname:
                    return _reply(client, 400, "only http:// URLs and CONNECT are proxied")
                host, port = url.hostname, url.port or 80
            if not host_allowed(host, self.allowed):
                self.refused.append((time.monotonic(), host))
                return _reply(client, 403, f"wren's sandbox doesn't allow connecting to {host}: it "
                                           "isn't in [sandbox] network")
            try:
                upstream = socket.create_connection((host, port), timeout=15)
            except OSError as e:
                return _reply(client, 502, f"can't connect to {host}:{port}: {e}")
            if method == b"CONNECT":
                client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                path = (url.path or "/") + (f"?{url.query}" if url.query else "")
                kept = [h for h in headers.split(b"\r\n") if h and not h.lower().startswith(
                    (b"proxy-", b"connection:", b"keep-alive:"))]
                # One request per connection, so a second one can't go to another host.
                upstream.sendall(b" ".join([method, path.encode("latin-1"), version]) + b"\r\n"
                                 + b"".join(h + b"\r\n" for h in kept)
                                 + b"Connection: close\r\n\r\n")
            if rest:
                upstream.sendall(rest)
            _pipe(client, upstream)
        except OSError:
            pass
        finally:
            client.close()
            if upstream is not None:
                upstream.close()


def _read_head(sock: socket.socket) -> tuple[bytes | None, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk or len(data) > MAX_HEAD:
            return None, b""
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def _split_host_port(target: str, default: int) -> tuple[str, int]:
    if target.startswith("["):  # [::1]:443
        host, _, port = target[1:].partition("]")
        port = port.lstrip(":")
    else:
        host, _, port = target.rpartition(":") if ":" in target else (target, "", "")
    return host, int(port) if port.isdigit() else default


def _reply(sock: socket.socket, status: int, message: str) -> None:
    reason = {400: "Bad Request", 403: "Forbidden", 502: "Bad Gateway"}[status]
    body = (message + "\n").encode()
    sock.sendall(f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\n"
                 f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)


def _pipe(a: socket.socket, b: socket.socket) -> None:
    """Copy both ways until both sides are done (or nothing moves for a long time)."""
    sel = selectors.DefaultSelector()
    peer = {a: b, b: a}
    for s in (a, b):
        s.setblocking(False)
        sel.register(s, selectors.EVENT_READ)
    open_sides = 2
    while open_sides:
        events = sel.select(IDLE_TIMEOUT)
        if not events:
            return
        for key, _ in events:
            src: socket.socket = key.fileobj  # type: ignore[assignment]
            try:
                data = src.recv(65536)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                data = b""
            if data:
                peer[src].setblocking(True)
                peer[src].sendall(data)
                peer[src].setblocking(False)
            else:
                sel.unregister(src)
                open_sides -= 1
                try:
                    peer[src].shutdown(socket.SHUT_WR)
                except OSError:
                    pass
