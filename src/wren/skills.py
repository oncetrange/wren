"""Skills: task-specific instructions and resources, loaded only when needed.

A skill is a directory holding SKILL.md (YAML frontmatter with `name` and
`description`, then markdown instructions) plus any scripts or references it
mentions. Only names and descriptions go into the system prompt; the model
loads a skill's full text with the `skill` tool when a task calls for it, and
the user can run one directly as `/<name> [arguments]`. This follows the Agent
Skills format, so skills written for Claude Code work unchanged.

Search order, later entries overriding earlier ones with the same name:
~/.claude/skills, <project>/.claude/skills, ~/.wren/skills, <project>/.wren/skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wren.config import CONFIG_DIR
from wren.frontmatter import FrontmatterError, read_frontmatter

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME, MAX_DESCRIPTION = 64, 1024
MAX_LISTED_FILES = 50
# Marks text wren adds to a user message; hidden in transcripts (like reminders).
SKILL_TAG = "<wren-skill"


class SkillError(Exception):
    pass


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # the SKILL.md file
    body: str
    source: str  # which directory it came from, for /skills
    model_invocable: bool = True
    argument_hint: str | None = None

    @property
    def dir(self) -> Path:
        return self.path.parent

    def files(self) -> list[str]:
        """Other files in the skill, relative to its directory."""
        found = sorted(str(p.relative_to(self.dir)) for p in self.dir.rglob("*")
                       if p.is_file() and p != self.path and not p.name.startswith("."))
        return found[:MAX_LISTED_FILES]

    def content(self) -> str:
        """The full skill as the model sees it when loaded."""
        files = self.files()
        listing = "\n".join(f"- {f}" for f in files) if files else "(none)"
        return (f'{SKILL_TAG} name="{self.name}">\n'
                f"Skill directory: {self.dir}\n"
                f"Files in it (paths relative to that directory):\n{listing}\n\n"
                f"{self.body.strip()}\n</wren-skill>")

    def invocation(self, arguments: str) -> str:
        """The skill as a user-invoked command: $ARGUMENTS substituted, or appended."""
        text = self.content()
        if "$ARGUMENTS" in text:
            return text.replace("$ARGUMENTS", arguments)
        return f"{text}\n\nArguments: {arguments}" if arguments else text


def parse_skill(path: Path, source: str) -> Skill:
    try:
        meta, body = read_frontmatter(path)
    except FrontmatterError as e:
        raise SkillError(str(e)) from None
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not NAME_RE.match(name) or len(name) > MAX_NAME:
        raise SkillError(f"{path}: 'name' must be lowercase letters, digits and hyphens "
                         f"(max {MAX_NAME}), got {name!r}")
    if name != path.parent.name:
        raise SkillError(f"{path}: name {name!r} must match its directory {path.parent.name!r}")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(f"{path}: 'description' is required")
    if len(description) > MAX_DESCRIPTION:
        raise SkillError(f"{path}: 'description' is longer than {MAX_DESCRIPTION} characters")
    hint = meta.get("argument-hint")
    if isinstance(hint, list):  # unquoted `[issue]` is a YAML list: keep it as written
        hint = "[" + " ".join(map(str, hint)) + "]"
    return Skill(
        name=name,
        description=" ".join(description.split()),
        path=path,
        body=body,
        source=source,
        model_invocable=not meta.get("disable-model-invocation", False),
        argument_hint=str(hint) if hint is not None else None,
    )


def skill_dirs(cwd: Path, home: Path | None = None) -> list[tuple[Path, str]]:
    """(directory, label) in increasing precedence."""
    home = home or CONFIG_DIR
    return [
        (Path.home() / ".claude" / "skills", "~/.claude/skills"),
        (cwd / ".claude" / "skills", ".claude/skills"),
        (home / "skills", "~/.wren/skills"),
        (cwd / ".wren" / "skills", ".wren/skills"),
    ]


def discover(cwd: Path, home: Path | None = None) -> tuple[dict[str, Skill], list[str]]:
    """All skills by name, and warnings about invalid or shadowed ones."""
    skills: dict[str, Skill] = {}
    warnings: list[str] = []
    for directory, label in skill_dirs(cwd, home):
        if not directory.is_dir():
            continue
        for skill_md in sorted(directory.glob("*/SKILL.md")):
            try:
                skill = parse_skill(skill_md, label)
            except (SkillError, OSError) as e:
                warnings.append(f"skipping skill: {e}")
                continue
            if skill.name in skills:
                warnings.append(f"skill {skill.name!r} from {label} overrides the one in "
                                f"{skills[skill.name].source}")
            skills[skill.name] = skill
    return skills, warnings


def prompt_section(skills: dict[str, Skill]) -> str:
    """The system prompt's list of skills the model may load."""
    listed = [s for s in skills.values() if s.model_invocable]
    if not listed:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in sorted(listed, key=lambda s: s.name))
    return ("# Skills\n"
            "Skills hold instructions and resources for particular tasks. When a task matches a "
            "skill's description, load it with the skill tool before starting and follow it.\n"
            f"{lines}\n")


def expand(text: str, skills: dict[str, Skill], reserved: set[str] = frozenset()) -> tuple[Skill, str] | None:
    """`/name arguments` -> (skill, arguments) if it names a skill (and not a reserved command)."""
    if not text.startswith("/"):
        return None
    name, _, arguments = text[1:].partition(" ")
    if name in reserved or name not in skills:
        return None
    return skills[name], arguments.strip()
