"""Markdown files with YAML frontmatter (skills, agent definitions)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)(.*)$", re.S)


class FrontmatterError(Exception):
    pass


def read_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    """(metadata, body). Raises FrontmatterError or OSError."""
    m = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if not m:
        raise FrontmatterError(f"{path}: must start with YAML frontmatter between --- lines")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise FrontmatterError(f"{path}: invalid frontmatter: {e}") from None
    if not isinstance(meta, dict):
        raise FrontmatterError(f"{path}: frontmatter must be a mapping")
    return meta, m.group(2)
