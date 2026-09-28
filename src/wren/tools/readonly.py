"""Recognize shell commands that only read: what an explore subagent may run.

Deliberately conservative: a command passes only if every program in it (split
at pipes, `&&`, `||` and `;`) is on the list below with no argument that makes
it write, and nothing redirects into a file or substitutes another command.
Anything else is refused, so a false "no" costs a retry, never a changed file.
"""

from __future__ import annotations

import re
import shlex

# Programs that never modify anything, whatever their arguments (except as noted).
SAFE_PROGRAMS = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "tree", "file",
    "stat", "pwd", "echo", "which", "type", "du", "sort", "uniq", "cut", "diff", "basename",
    "dirname", "realpath", "readlink", "nl", "tr", "cd", "true", "test", "[",
})
# Arguments that turn an otherwise read-only program into one that writes or runs things.
UNSAFE_ARGS = {
    "find": ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"),
    "sort": ("-o", "--output"),
    "rg": ("--pre",),
    "tree": ("-o",),
}
SAFE_GIT = frozenset({
    "status", "log", "diff", "show", "blame", "ls-files", "ls-tree", "grep", "rev-parse",
    "describe", "shortlog", "cat-file", "merge-base", "reflog",
})
# Harmless redirections, removed before checking what is left.
_NULL_REDIRECT = re.compile(r"\s*(?:\d?>>?|&>)\s*/dev/null|\s*\d?>&\d")
_SEPARATORS = {"|", "||", "&&", ";"}


def is_read_only_command(command: str) -> bool:
    command = _NULL_REDIRECT.sub(" ", command)
    if "\n" in command or "`" in command or "$(" in command or "<(" in command:
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    segment: list[str] = []
    for token in tokens + [";"]:
        if token in _SEPARATORS:
            if not segment or not _safe_segment(segment):
                return False
            segment = []
        elif token and set(token) <= set("();<>|&"):
            return False  # a redirection, background job or subshell
        else:
            segment.append(token)
    return True


def _safe_segment(args: list[str]) -> bool:
    program, rest = args[0], args[1:]
    if "=" in program:
        return False  # VAR=value prefixes can change what runs
    if program == "git":
        sub = next((a for a in rest if not a.startswith("-")), None)
        return sub in SAFE_GIT and not any(a.startswith("--output") for a in rest)
    if program not in SAFE_PROGRAMS:
        return False
    unsafe = UNSAFE_ARGS.get(program, ())
    return not any(a == u or a.startswith(u + "=") or (u == "-o" and re.match(r"-[^-]*o", a))
                   for a in rest for u in unsafe)
