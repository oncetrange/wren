import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RecordingUI, ScriptedProvider, reply
from rich.console import Console

from wren.agent.loop import Agent
from wren.agent.permissions import Permissions
from wren.agent.plans import WREN_DIR_GITIGNORE
from wren.cli.main import BUILTIN_NAMES, run_prompt
from wren.cli.ui import RichUI
from wren.config import ModelConfig
from wren.llm.types import Message, Response, ToolUseBlock
from wren.skills import SkillError, discover, expand, parse_skill, prompt_section
from wren.tools import ToolContext


def write_skill(root: Path, name: str, description="Does a thing. Use when asked.", body="Step 1.",
                extra="", dirname=None):
    d = root / (dirname or name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n")
    return d


def test_parse(tmp_path):
    d = write_skill(tmp_path, "commit-message", description=">\n  Write commit messages.\n  Use when committing.",
                    extra="argument-hint: [issue]\n", body="# Rules\n- Subject <= 60 chars")
    (d / "scripts").mkdir()
    (d / "scripts" / "check.sh").write_text("true")
    skill = parse_skill(d / "SKILL.md", "test")
    assert skill.description == "Write commit messages. Use when committing."
    assert skill.argument_hint == "[issue]" and skill.model_invocable
    assert skill.files() == ["scripts/check.sh"]
    content = skill.content()
    assert content.startswith('<wren-skill name="commit-message">')
    assert str(d) in content and "- scripts/check.sh" in content and "Subject <= 60 chars" in content


@pytest.mark.parametrize("name,dirname,description,match", [
    ("Bad_Name", None, "x", "lowercase"),
    ("good", "other", "x", "must match its directory"),
    ("good", None, "''", "description"),
])
def test_invalid(tmp_path, name, dirname, description, match):
    d = write_skill(tmp_path, name, description=description, dirname=dirname or name.lower())
    with pytest.raises(SkillError, match=match):
        parse_skill(d / "SKILL.md", "test")


def test_no_frontmatter(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "SKILL.md").write_text("# just markdown")
    with pytest.raises(SkillError, match="frontmatter"):
        parse_skill(tmp_path / "x" / "SKILL.md", "test")


def test_arguments(tmp_path):
    with_placeholder = parse_skill(write_skill(tmp_path, "a", body="Release version $ARGUMENTS.") / "SKILL.md", "t")
    assert "Release version 1.2.0." in with_placeholder.invocation("1.2.0")
    plain = parse_skill(write_skill(tmp_path, "b") / "SKILL.md", "t")
    assert plain.invocation("see #12").endswith("Arguments: see #12")
    assert "Arguments" not in plain.invocation("")


def test_discovery_precedence_and_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "userhome")
    project, home = tmp_path / "project", tmp_path / "wrenhome"
    write_skill(tmp_path / "userhome" / ".claude" / "skills", "shared", description="from claude")
    write_skill(home / "skills", "shared", description="from wren user")
    write_skill(project / ".wren" / "skills", "shared", description="from project")
    write_skill(home / "skills", "manual", extra="disable-model-invocation: true\n")
    (project / ".wren" / "skills" / "broken").mkdir()
    (project / ".wren" / "skills" / "broken" / "SKILL.md").write_text("no frontmatter")

    skills, warnings = discover(project, home)
    assert skills["shared"].description == "from project"
    assert skills["shared"].source == ".wren/skills"
    assert any("overrides" in w for w in warnings) and any("broken" in w for w in warnings)
    section = prompt_section(skills)
    assert "- shared: from project" in section and "manual" not in section


# --- invocation ------------------------------------------------------------------

def load(skill_name, id="t1"):
    return Response(Message("assistant", [ToolUseBlock(id, "skill", {"name": skill_name})]), "tool_use")


@pytest.fixture
def skills(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "release", description="Cut a release. Use when asked to release.",
                body="Bump the version to $ARGUMENTS, then tag it.")
    write_skill(root, "deploy", extra="disable-model-invocation: true\n", body="Deploy steps.")
    return {s.name: s for s in (parse_skill(root / n / "SKILL.md", "test") for n in ("release", "deploy"))}


def agent_with(skills, turns, tmp_path):
    (tmp_path / "p").mkdir(exist_ok=True)
    return Agent(ScriptedProvider(turns), ModelConfig(name="fake", model="f"),
                 ToolContext(cwd=(tmp_path / "p").resolve()), RecordingUI(),
                 Permissions(mode="auto"), skills=skills)


def test_model_loads_listed_skills(skills, tmp_path):
    agent = agent_with(skills, [load("release"), load("deploy", "t2"), reply("ok")], tmp_path)
    assert "- release: Cut a release." in agent.system and "deploy" not in agent.system
    agent.run("ship it")
    loaded = agent.provider.requests[1][-1].content[0]
    assert "Bump the version to $ARGUMENTS" in loaded.content and not loaded.is_error
    manual = agent.provider.requests[2][-1].content[0]
    assert manual.is_error and "no skill named 'deploy'" in manual.content


def test_no_skill_tool_without_skills(tmp_path):
    agent = agent_with({}, [], tmp_path)
    assert "skill" not in agent.tools and "# Skills" not in agent.system


def test_user_invocation(skills, tmp_path):
    agent = agent_with(skills, [reply("released"), reply("deployed")], tmp_path)
    run_prompt(agent, "/release 1.2.0")
    first = agent.messages[0]
    assert first.content[0].text == "/release 1.2.0"
    assert "Bump the version to 1.2.0, then tag it." in first.content[1].text
    run_prompt(agent, "/deploy")                    # manual-only skills work for the user
    assert "Deploy steps." in agent.messages[2].content[1].text

    out = io.StringIO()
    RichUI(Console(file=out, width=80, color_system=None), interactive=False).render_history(
        agent.messages, agent.tools, agent.ctx)
    assert "/release 1.2.0" in out.getvalue() and "Bump the version" not in out.getvalue()


def test_builtin_commands_win(skills):
    skills = dict(skills, help=skills["release"])
    assert expand("/help", skills, BUILTIN_NAMES) is None
    assert expand("/release now", skills, BUILTIN_NAMES)[1] == "now"
    assert expand("release", skills, BUILTIN_NAMES) is None


def test_slash_menu_matches():
    from wren.cli.completion import matches
    entries = [("/release", "cut a release"), ("/resume", "switch"), ("/help", "")]

    def names(text):
        return [n for n, _ in matches(text, entries)]
    assert names("/re") == ["/release", "/resume"]
    assert names("/release 1") == [] and names("hello") == [] and names("") == []


def test_headless_skill(tmp_path):
    from fake_anthropic import FakeAnthropic, text_turn
    home, project = tmp_path / "home", tmp_path / "project"
    write_skill(home / "skills", "release", body="Bump the version to $ARGUMENTS.")
    project.mkdir()
    with FakeAnthropic([text_turn("done")]) as server:
        (home / "config.toml").write_text(
            f'[models.fake]\nmodel = "f"\nbase_url = "{server.url}"\napi_key_env = "K"\nprompt_cache = false\n')
        subprocess.run([sys.executable, "-m", "wren.cli.main", "-p", "/release 2.0", "-m", "fake"],
                       cwd=project, env={**os.environ, "WREN_HOME": str(home), "K": "k"},
                       capture_output=True, text=True, timeout=60, check=True)
        body = server.requests[0]["body"]
    assert "Bump the version to 2.0." in body["messages"][0]["content"][1]["text"]
    assert "- release:" in body["system"] and any(t["name"] == "skill" for t in body["tools"])


def test_wren_dir_gitignore_shares_hooks_and_skills(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    wren = tmp_path / ".wren"
    (wren / "plans").mkdir(parents=True)
    (wren / ".gitignore").write_text(WREN_DIR_GITIGNORE)
    (wren / "plans" / "p.md").write_text("plan")
    (wren / "hooks.toml").write_text("")
    write_skill(wren / "skills", "x")
    status = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=tmp_path,
                            capture_output=True, text=True).stdout
    assert ".wren/hooks.toml" in status and ".wren/skills/x/SKILL.md" in status
    assert "plans" not in status and ".gitignore" not in status


def test_example_skills_are_valid():
    root = Path(__file__).parents[1] / "examples" / "skills"
    for skill_md in root.glob("*/SKILL.md"):
        skill = parse_skill(skill_md, "examples")
        assert skill.files() and skill.description
