# Skills

A skill packages instructions (and any scripts or references they need) for a particular task,
such as your team's commit format or a release checklist. Only each skill's name and description
sit in the model's context; it loads the full instructions with the `skill` tool when a task
matches, so you can keep many skills around at no cost. Run one yourself with `/<name> [arguments]`
(Tab completes it), in a session or headless: `wren -p "/release 1.2.0"`.

```
~/.wren/skills/<name>/SKILL.md      # yours, in every project
.wren/skills/<name>/SKILL.md        # the project's, shareable in git
```

```markdown
---
name: release
description: Cut a release. Use when asked to release or publish a new version.
argument-hint: "[version]"
disable-model-invocation: true    # optional: only run when the user types /release
---
Bump the version to $ARGUMENTS, update CHANGELOG.md, run scripts/check.sh, then tag.
```

Skills use the Agent Skills format, so `~/.claude/skills` and `.claude/skills` are read too (wren's
own directories win on name clashes). `/skills` lists them. See `examples/skills/commit-message`
for a complete one with a helper script.
