from pathlib import Path

import pytest

from wren.skills import SkillError, discover, parse_skill, prompt_section


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
