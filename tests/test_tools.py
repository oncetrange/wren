import pytest

from wren.tools import ToolContext, ToolError
from wren.tools.base import validate_args
from wren.tools.files import EditFile, ReadFile, WriteFile
from wren.tools.search import Glob, Grep
from wren.tools.shell import Bash


def test_read_numbers_lines_and_pages(ctx: ToolContext):
    (ctx.cwd / "a.txt").write_text("one\ntwo\nthree\n")
    out = ReadFile().run({"path": "a.txt", "offset": 2, "limit": 1}, ctx)
    assert out.content.startswith("     2\ttwo")
    assert "showing lines 2-2 of 3" in out.content


def test_read_rejects_binary(ctx: ToolContext):
    (ctx.cwd / "b.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        ReadFile().run({"path": "b.bin"}, ctx)


def test_edit_requires_prior_read(ctx: ToolContext):
    (ctx.cwd / "a.py").write_text("x = 1\n")
    with pytest.raises(ToolError, match="read a.py"):
        EditFile().run({"path": "a.py", "old_string": "1", "new_string": "2"}, ctx)


def test_edit_replaces_unique_match(ctx: ToolContext):
    f = ctx.cwd / "a.py"
    f.write_text("x = 1\ny = 2\n")
    ReadFile().run({"path": "a.py"}, ctx)
    out = EditFile().run({"path": "a.py", "old_string": "y = 2", "new_string": "y = 3"}, ctx)
    assert f.read_text() == "x = 1\ny = 3\n"
    assert out.summary == "+1 -1"


def test_edit_rejects_ambiguous_match(ctx: ToolContext):
    f = ctx.cwd / "a.py"
    f.write_text("a = 0\na = 0\n")
    ReadFile().run({"path": "a.py"}, ctx)
    with pytest.raises(ToolError, match="occurs 2 times"):
        EditFile().run({"path": "a.py", "old_string": "a = 0", "new_string": "a = 1"}, ctx)
    EditFile().run({"path": "a.py", "old_string": "a = 0", "new_string": "a = 1", "replace_all": True}, ctx)
    assert f.read_text() == "a = 1\na = 1\n"


def test_edit_detects_external_change(ctx: ToolContext):
    f = ctx.cwd / "a.py"
    f.write_text("x = 1\n")
    ReadFile().run({"path": "a.py"}, ctx)
    ctx.read_files[f] -= 1  # simulate the file changing on disk after the read
    with pytest.raises(ToolError, match="changed on disk"):
        EditFile().run({"path": "a.py", "old_string": "1", "new_string": "2"}, ctx)


def test_edit_preserves_crlf(ctx: ToolContext):
    f = ctx.cwd / "w.txt"
    f.write_bytes(b"a\r\nb\r\n")
    ReadFile().run({"path": "w.txt"}, ctx)
    EditFile().run({"path": "w.txt", "old_string": "a\nb", "new_string": "a\nc"}, ctx)
    assert f.read_bytes() == b"a\r\nc\r\n"


def test_write_creates_dirs_and_guards_overwrite(ctx: ToolContext):
    WriteFile().run({"path": "pkg/new.py", "content": "print(1)\n"}, ctx)
    assert (ctx.cwd / "pkg/new.py").read_text() == "print(1)\n"
    (ctx.cwd / "old.py").write_text("keep\n")
    with pytest.raises(ToolError, match="before modifying"):
        WriteFile().run({"path": "old.py", "content": "gone\n"}, ctx)


def test_bash_reports_exit_code(ctx: ToolContext):
    ok = Bash().run({"command": "echo hi"}, ctx)
    assert ok.content == "hi\n\n[exit code 0]" and not ok.is_error
    bad = Bash().run({"command": "echo oops >&2; exit 3"}, ctx)
    assert bad.is_error and "oops" in bad.content and "[exit code 3]" in bad.content


def test_bash_timeout(ctx: ToolContext):
    with pytest.raises(ToolError, match="timed out"):
        Bash().run({"command": "sleep 5", "timeout": 1}, ctx)


def test_bash_permission_key():
    assert Bash().permission_key({"command": "pytest -x tests"}) == "bash:pytest"
    assert Bash().permission_key({"command": "pytest && rm -rf /"}) == "bash:pytest && rm -rf /"


def test_glob_and_grep(ctx: ToolContext):
    (ctx.cwd / "src").mkdir()
    (ctx.cwd / "src/app.py").write_text("def main():\n    return 42\n")
    (ctx.cwd / "node_modules").mkdir()
    (ctx.cwd / "node_modules/x.py").write_text("def main(): pass\n")
    assert Glob().run({"pattern": "**/*.py"}, ctx).content == "src/app.py"
    out = Grep().run({"pattern": r"def \w+", "glob": "*.py"}, ctx)
    assert out.content.startswith("src/app.py:1:def main")
    assert "node_modules" not in out.content


def test_validate_args():
    schema = ReadFile.input_schema
    assert validate_args(schema, {"path": "a"}) is None
    assert "missing" in validate_args(schema, {})
    assert "unknown argument" in validate_args(schema, {"path": "a", "bogus": 1})
    assert "integer" in validate_args(schema, {"path": "a", "limit": "10"})
    assert "integer" in validate_args(schema, {"path": "a", "limit": True})
