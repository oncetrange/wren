"""Inside a Linux sandbox, reach the proxy: `python -m wren.netbridge SOCKET PORT -- CMD...`

bwrap gives the command a network namespace of its own, where nothing outside
is reachable, not even the host's localhost. This listens on 127.0.0.1:PORT in
that namespace, passes each connection on to the proxy's unix socket (bound in
from outside), runs CMD and exits with its status.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import threading

from wren.netproxy import _pipe


def main(argv: list[str]) -> int:
    path, port, sep, *command = argv
    if sep != "--" or not command:
        print("usage: python -m wren.netbridge SOCKET PORT -- COMMAND...", file=sys.stderr)
        return 2
    server = socket.create_server(("127.0.0.1", int(port)))
    threading.Thread(target=_accept, args=(server, path), daemon=True).start()
    proc = subprocess.Popen(command)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _: proc.send_signal(s))
    return proc.wait()


def _accept(server: socket.socket, path: str) -> None:
    while True:
        client, _ = server.accept()
        threading.Thread(target=_forward, args=(client, path), daemon=True).start()


def _forward(client: socket.socket, path: str) -> None:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
        _pipe(client, upstream)
    except OSError:
        pass
    finally:
        client.close()
        upstream.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
