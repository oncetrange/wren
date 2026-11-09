from wren.tools.base import MAX_OUTPUT_CHARS, truncate


def test_short_output_untouched():
    assert truncate("abc") == "abc"


def test_truncate_keeps_tail_and_error_lines():
    middle = "\n".join(f"line {i}" for i in range(5000))
    text = "start\n" + middle + "\nFAILED test_x - AssertionError\n" + middle + "\n3 failed, 9 passed"
    out = truncate(text)
    assert out.startswith("start")
    assert out.endswith("3 failed, 9 passed")
    assert "FAILED test_x - AssertionError" in out
    assert len(out) <= MAX_OUTPUT_CHARS + 500


def test_truncate_without_signal_lines():
    out = truncate("x" * 100_000)
    assert "characters truncated]" in out
    assert len(out) <= MAX_OUTPUT_CHARS + 100
