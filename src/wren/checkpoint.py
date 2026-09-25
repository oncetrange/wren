"""Workspace snapshots in a shadow git repository.

Each project gets its own bare-ish repository under ~/.wren/checkpoints whose
work tree is the project directory. The project's own .git is never touched,
so checkpoints work in non-git projects and don't pollute real history.
Because a snapshot captures the whole tree (respecting .gitignore), changes
made through `bash` are covered as well as those from the file tools.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from wren.config import CONFIG_DIR

CHECKPOINTS_DIR = CONFIG_DIR / "checkpoints"
# Always excluded, on top of the project's own .gitignore files.
DEFAULT_EXCLUDES = """\
.git/
node_modules/
.venv/
venv/
__pycache__/
*.pyc
.mypy_cache/
.pytest_cache/
.ruff_cache/
.tox/
.DS_Store
"""
_IDENTITY = {
    "GIT_AUTHOR_NAME": "wren", "GIT_AUTHOR_EMAIL": "wren@localhost",
    "GIT_COMMITTER_NAME": "wren", "GIT_COMMITTER_EMAIL": "wren@localhost",
}


class CheckpointError(Exception):
    pass


class Checkpoints:
    def __init__(self, work_tree: Path, root: Path = CHECKPOINTS_DIR):
        self.work_tree = work_tree
        key = hashlib.sha256(str(work_tree).encode()).hexdigest()[:16]
        self.git_dir = root / key
        self._head: str | None = None
        self.disabled_reason = self._check_usable()

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def snapshot(self, label: str) -> str:
        """Record the current workspace state; returns the snapshot id."""
        tree = self._write_tree()
        args = ["commit-tree", tree, "-m", label[:200]] + (["-p", self._head] if self._head else [])
        self._head = self._git(*args).strip()
        # Keep snapshots reachable so git never garbage-collects them.
        self._git("update-ref", "refs/heads/wren", self._head)
        return self._head

    def changed_files(self, commit: str) -> list[str]:
        """Files that differ between a snapshot and the workspace right now,
        as "<A|D|M> <path>" relative to the snapshot."""
        current = self._write_tree()
        out = self._git("diff", "--name-status", "--no-renames", commit, current)
        return [line.replace("\t", " ", 1) for line in out.splitlines()]

    def restore(self, commit: str) -> None:
        """Make the workspace match a snapshot: modified files are reverted,
        files created since are deleted, deleted files come back."""
        current = self._write_tree()
        created = self._git("diff", "--name-only", "-z", "--no-renames", "--diff-filter=A",
                            commit, current).split("\0")
        # Point the index at the snapshot and write every file in it back out.
        self._git("read-tree", commit)
        self._git("checkout-index", "--all", "--force")
        for rel in filter(None, created):
            path = self.work_tree / rel
            path.unlink(missing_ok=True)
            _remove_empty_parents(path.parent, self.work_tree)

    # --- internals ---------------------------------------------------------

    def _check_usable(self) -> str | None:
        if shutil.which("git") is None:
            return "git is not installed"
        if self.work_tree in (Path.home(), Path(self.work_tree.anchor)):
            return f"refusing to snapshot {self.work_tree}"
        try:
            if not (self.git_dir / "HEAD").exists():
                self.git_dir.mkdir(parents=True, exist_ok=True)
                self._git("init", "--quiet")
                self._git("config", "core.autocrlf", "false")
                self._git("config", "gc.auto", "0")
                (self.git_dir / "info").mkdir(exist_ok=True)
                (self.git_dir / "info" / "exclude").write_text(
                    DEFAULT_EXCLUDES + self._own_dirs_excludes()
                )
        except CheckpointError as e:
            return str(e)
        return None

    def _own_dirs_excludes(self) -> str:
        """Never snapshot our own state (shadow repo, sessions) if it lives in the work tree."""
        lines = []
        for d in (self.git_dir, CONFIG_DIR):
            try:
                lines.append(f"/{d.resolve().relative_to(self.work_tree).as_posix()}/\n")
            except ValueError:
                pass
        return "".join(lines)

    def _write_tree(self) -> str:
        self._git("add", "-A", ".")
        return self._git("write-tree").strip()

    def _git(self, *args: str) -> str:
        env = {**os.environ, **_IDENTITY, "GIT_DIR": str(self.git_dir),
               "GIT_WORK_TREE": str(self.work_tree)}
        env.pop("GIT_INDEX_FILE", None)
        try:
            proc = subprocess.run(["git", *args], cwd=self.work_tree, env=env,
                                  capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CheckpointError(f"git {args[0]} failed: {e}") from e
        if proc.returncode != 0:
            raise CheckpointError(f"git {args[0]} failed: {proc.stderr.strip()}")
        return proc.stdout


def _remove_empty_parents(directory: Path, stop: Path) -> None:
    while directory != stop and directory.is_relative_to(stop):
        try:
            directory.rmdir()
        except OSError:
            return  # not empty
        directory = directory.parent
