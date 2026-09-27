import pytest

from wren.tools import ToolContext, ToolError
from wren.tools.files import EditFile, ReadFile

GO = (
    "func main() {\n"
    "\tif ok {\n"
    "\t\t// require(\"x\")\n"
    "\t\tload(x)\n"
    "\n"
    "\t\treturn evaluated\n"
    "\t}\n"
    "}\n"
)


def edit(ctx, text, old, new, name="f.go"):
    f = ctx.cwd / name
    f.write_bytes(text.encode()) if isinstance(text, str) else f.write_bytes(text)
    ReadFile().run({"path": name}, ctx)
    out = EditFile().run({"path": name, "old_string": old, "new_string": new}, ctx)
    return f.read_bytes().decode(), out


@pytest.mark.parametrize("extra,missing", [("\t", ""), ("", "\t")])
def test_tab_depth_off_by_one_is_reindented(ctx: ToolContext, extra, missing):
    # the model's block is one tab deeper / shallower than the file
    def shift(block):
        return "\n".join(extra + l.removeprefix(missing) if l.strip() else l for l in block.split("\n"))
    old = shift("\t\t// require(\"x\")\n\t\tload(x)\n\n\t\treturn evaluated")
    new = shift("\t\t// require(\"x\")\n\t\tload(x)\n\t\tcache(x)\n\n\t\treturn evaluated")
    result, out = edit(ctx, GO, old, new)
    assert result == GO.replace("\t\tload(x)\n", "\t\tload(x)\n\t\tcache(x)\n")
    assert "re-indented" in out.content


def test_nested_new_lines_keep_relative_indent(ctx: ToolContext):
    py = "class A:\n    def f(self):\n        return 1\n"
    old = "def f(self):\n    return 1"          # model dropped the class-level indent
    new = "def f(self):\n    if self:\n        return 2\n    return 1"
    result, _ = edit(ctx, py, old, new, "a.py")
    assert result == "class A:\n    def f(self):\n        if self:\n            return 2\n        return 1\n"


def test_trailing_whitespace_and_crlf(ctx: ToolContext):
    crlf = GO.replace("\n", "\r\n").encode()
    result, _ = edit(ctx, crlf, "\t\t\tload(x)   ", "\t\t\tload(y)")
    assert "\t\tload(y)\r\n" in result and "\n" not in result.replace("\r\n", "")


def test_inconsistent_indentation_is_not_guessed(ctx: ToolContext):
    # first line one tab deeper, second line two tabs deeper: no single shift
    with pytest.raises(ToolError, match="most similar part of the file is around line 3"):
        edit(ctx, GO, "\t\t\t// require(\"x\")\n\t\t\t\tload(x)", "x")
    assert (ctx.cwd / "f.go").read_text() == GO


def test_loose_ambiguity_is_reported(ctx: ToolContext):
    text = "a:\n\tx = 1\nb:\n\tx = 1\n"
    with pytest.raises(ToolError, match="matches 2 places \\(lines 2, 4\\)"):
        edit(ctx, text, "\t\tx = 1", "\t\tx = 2", "t.txt")


def test_not_found_shows_the_closest_region(ctx: ToolContext):
    with pytest.raises(ToolError) as e:
        edit(ctx, GO, "\t\tload(xs)\n", "\t\tload(y)\n")
    assert "around line 4" in str(e.value) and "     4→\t\tload(x)" in str(e.value)


def test_exact_matches_are_unchanged(ctx: ToolContext):
    result, out = edit(ctx, GO, "\t\tload(x)", "\t\tload(y)")
    assert result == GO.replace("load(x)", "load(y)") and "re-indented" not in out.content
