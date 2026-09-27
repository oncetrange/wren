"""Layered context management: masking (L1), anchored summaries (L2), archives (L3)."""

import pytest

from wren.agent import compact, loop
from wren.agent.compact import KEEP_TURNS, mask_old_tool_traffic, recent_start
from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.session import SessionLog, load_session
from wren.config import ModelConfig, Price
from wren.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage

from conftest import RecordingUI, ScriptedProvider, call, reply
from wren.agent.compact import estimate_tokens
from wren.llm.types import Completed


class SizedProvider(ScriptedProvider):
    """Reports prompt sizes like a real API does, so thresholds trigger, and
    answers compaction requests with "SUMMARY <n>" whenever they come."""

    summaries = 0

    def stream(self, *, system, messages, tools):
        last = messages[-1].content[-1]
        if isinstance(last, TextBlock) and "about to be compacted" in last.text:
            self.summaries += 1
            self.requests.append(messages)
            yield Completed(reply(f"SUMMARY {self.summaries}"))
            return
        for event in super().stream(system=system, messages=messages, tools=tools):
            if isinstance(event, Completed):
                event.response.usage = Usage(
                    input_tokens=estimate_tokens(messages) + len(system) // 4, output_tokens=5)
            yield event


def exchange(i: int, size: int = 2000) -> list[Message]:
    return [
        Message("assistant", [ToolUseBlock(f"t{i}", "read_file", {"path": f"f{i}.py"})]),
        Message("user", [ToolResultBlock(f"t{i}", "x" * size)]),
    ]


def history(turns: int, size: int = 2000) -> list[Message]:
    msgs = [Message("user", [TextBlock("task")])]
    for i in range(turns):
        msgs += exchange(i, size)
    return msgs


def results(msgs):
    return [b for m in msgs for b in m.content if isinstance(b, ToolResultBlock)]


def test_masks_only_old_large_outputs():
    msgs = history(12)
    msgs[2].content.append(ToolResultBlock("small", "ok"))
    masked, freed = mask_old_tool_traffic(msgs)
    out = [r for r in results(masked) if r.tool_use_id != "small"]
    assert all(r.content.startswith("[output omitted") for r in out[:4])       # turns 0-3
    assert "read_file call" in out[0].content
    assert next(r for r in results(masked) if r.tool_use_id == "small").content == "ok"
    assert all(r.content == "x" * 2000 for r in out[-KEEP_TURNS:])              # recent: kept
    assert freed > 4 * 1800
    assert results(msgs)[0].content == "x" * 2000                               # input untouched
    assert mask_old_tool_traffic(masked)[1] == 0                                # idempotent


def test_never_masks_the_models_own_tool_arguments():
    # Masked arguments get imitated: the model copies the placeholder into new edits.
    msgs = [Message("user", [TextBlock("t")]),
            Message("assistant", [ToolUseBlock("e", "write_file", {"path": "a.py", "content": "y" * 3000})]),
            Message("user", [ToolResultBlock("e", "created a.py")])] + history(KEEP_TURNS)[1:]
    masked, _ = mask_old_tool_traffic(msgs)
    assert masked[1].content[0].input["content"] == "y" * 3000


def test_small_gains_are_not_worth_a_cache_miss():
    from wren.agent.compact import plan_mask
    msgs = history(KEEP_TURNS + 1, size=500)         # only one small old output
    assert plan_mask(msgs, current_tokens=50_000, mask_at=40_000)[1] == 0


def test_recent_start_is_an_assistant_turn():
    msgs = history(10)
    i = recent_start(msgs)
    assert msgs[i].role == "assistant" and sum(m.role == "assistant" for m in msgs[i:]) == KEEP_TURNS
    assert recent_start(history(3)) == 0


# --- through the agent ---------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "p").mkdir()
    for i in range(30):  # read_file output ~4k chars, ~1k tokens each
        (tmp_path / "p" / f"f{i}.py").write_text(f"# file {i}\n" + "x = 1\n" * 300)
    monkeypatch.setattr(loop, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    return tmp_path


def make_agent(project, turns, **model):
    from wren.tools import ToolContext
    cfg = ModelConfig(name="fake", model="fake-1", **model)
    return Agent(SizedProvider(turns), cfg, ToolContext(cwd=(project / "p").resolve()),
                 RecordingUI(), Permissions(mode="auto"),
                 log=SessionLog(directory=project / "sessions"))


def reading(n):
    return [call("read_file", f"t{i}", path=f"f{i}.py") for i in range(n)]


def test_agent_masks_in_batches_and_replay_matches(project):
    agent = make_agent(project, reading(24) + [reply("done")], mask_at=12_000)
    agent.run("read everything")
    masks = [e for e in agent.ui.events if e[0] == "notice" and "cleared" in e[1]]
    assert 1 <= len(masks) <= 5                     # batched: not on each of 24 turns
    assert agent.estimated_context() < 12_000 + 1500
    assert [m.to_dict() for m in load_session(agent.log.path).messages] == \
        [m.to_dict() for m in agent.messages]


def test_compaction_keeps_recent_turns_and_archives(project):
    agent = make_agent(project, reading(12) + [reply("done")], mask_at=0, compact_at=10_000)
    agent.run("read everything")

    note = agent.messages[0].text()
    assert "SUMMARY 1" in note
    archive = project / "transcripts" / f"{agent.log.id}.md"
    assert str(archive) in note and "# Compaction 1" in archive.read_text()
    assert "## tool call: read_file" in archive.read_text()
    assert agent.provider.summaries == 1
    assert agent.messages[1].role == "assistant"     # recent turns kept verbatim after the note
    assert sum(m.role == "assistant" for m in agent.messages) >= 2

    state = load_session(agent.log.path)
    assert [m.to_dict() for m in state.messages] == [m.to_dict() for m in agent.messages]


def test_second_compaction_updates_the_summary(project):
    agent = make_agent(project, reading(24) + [reply("done")], mask_at=0, compact_at=10_000)
    agent.run("read everything")
    assert 2 <= agent.provider.summaries <= 5       # batched: not on every turn
    second = next(r for r in agent.provider.requests
                  if "SUMMARY 1" in getattr(r[0].content[0], "text", "")
                  and "compacted" in getattr(r[-1].content[-1], "text", ""))
    assert "produce an updated version" in second[-1].content[-1].text   # anchored on the old one
    assert f"SUMMARY {agent.provider.summaries}" in agent.messages[0].text()
    archive = (project / "transcripts" / f"{agent.log.id}.md").read_text()
    assert archive.count("# Compaction") == agent.provider.summaries
    assert "<summary>" not in archive                # earlier summaries aren't re-archived


def test_undo_compaction_restores_head_and_keeps_tail(project):
    agent = make_agent(project, reading(12) + [reply("done")], mask_at=0, compact_at=10_000)
    agent.run("read everything")
    index = next(i for i, p in enumerate(agent.conv.timeline) if p.kind == "compaction")
    agent.rewind(index)
    texts = [b.text for m in agent.messages for b in m.content if isinstance(b, TextBlock)]
    assert texts[0] == "read everything" and texts[-1] == "done"
    assert not any("SUMMARY" in t for t in texts)
    assert len(results(agent.messages)) == 12


def test_tiered_price():
    p = Price(tiers=[Price(up_to=32_000, input=1.0, output=5.0, cache_read=0.2),
                     Price(input=2.0, output=10.0, cache_read=0.4)])
    small = Usage(input_tokens=10_000, output_tokens=1000)
    large = Usage(input_tokens=1000, cache_read_tokens=60_000, output_tokens=1000)
    assert p.cost(small) == pytest.approx((10_000 * 1.0 + 1000 * 5.0) / 1e6)
    assert p.cost(large) == pytest.approx((1000 * 2.0 + 60_000 * 0.4 + 1000 * 10.0) / 1e6)
