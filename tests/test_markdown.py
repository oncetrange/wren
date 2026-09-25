import io

from rich.console import Console

from wren.cli.markdown import MarkdownStream


def feed(text: str, terminal: bool = True) -> tuple[MarkdownStream, str]:
    out = io.StringIO()
    stream = MarkdownStream(Console(file=out, force_terminal=terminal, width=60, color_system=None))
    printed: list[str] = []
    stream._flush_orig = stream._flush

    def spy():
        printed.append("\n".join(stream._block))
        stream._flush_orig()

    stream._flush = spy
    for i in range(0, len(text), 5):
        stream.feed(text[i : i + 5])
    stream.close()
    return printed, out.getvalue()


def test_splits_on_blank_lines_but_keeps_code_blocks_whole():
    blocks, _ = feed("Intro **bold**\n\n```py\na = 1\n\nb = 2\n```\nafter\n\n- x\n- y")
    blocks = [b for b in blocks if b]
    assert blocks == ["Intro **bold**", "```py\na = 1\n\nb = 2\n```", "after", "- x\n- y"]


def test_unterminated_fence_is_flushed_on_close():
    blocks, _ = feed("```\ncode")
    assert [b for b in blocks if b] == ["```\ncode"]


def test_plain_passthrough_when_not_a_terminal():
    _, out = feed("# Title\n\ntext", terminal=False)
    assert out == "# Title\n\ntext\n"
