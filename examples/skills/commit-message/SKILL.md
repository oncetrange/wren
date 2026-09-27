---
name: commit-message
description: Write git commit messages in a consistent format. Use when committing changes or when asked for a commit message.
argument-hint: "[extra context]"
---
# Commit messages

1. Look at what is staged (`git diff --cached --stat`, then the diff itself). If nothing is
   staged, stage only the files that belong to the change; never `git add .` blindly.
2. Write the message:
   - Subject: imperative mood ("Add", "Fix", not "Added"), at most 60 characters, no
     trailing period. Say what changes for the user or the code, not how you got there.
   - Blank line, then a body wrapped at 72 characters explaining why the change is needed
     and anything a reviewer should know. Skip the body for trivial changes.
   - Reference issues as "Fixes #123" on the last line when the user mentions one.
3. Check it with `scripts/check_message.sh` (in this skill's directory) before committing:
   `echo "$MESSAGE" | bash <skill dir>/scripts/check_message.sh`. Fix anything it reports.
4. Commit with the message and show the resulting `git log -1 --stat`.

Extra context from the user, if any: $ARGUMENTS
