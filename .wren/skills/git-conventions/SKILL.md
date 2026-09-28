---
name: git-conventions
description: Branch naming and commit message conventions for this repository. Use when creating a branch, committing, writing a commit message, or preparing changes for a pull request.
argument-hint: "[what the change is about]"
---
# Git conventions

## Types

Every branch and every commit has one type:

| type       | use for                                  |
|------------|------------------------------------------|
| `feat`     | a new feature                            |
| `fix`      | a bug fix                                |
| `perf`     | a performance change                     |
| `refactor` | restructuring without behavior change    |
| `revert`   | reverting an earlier change              |
| `chore`    | tooling, build, CI, dependencies, config |
| `test`     | tests only                               |
| `docs`     | documentation only                       |

Pick the type from what the change does for users of the code, not from which files it
touches: a bug fix that also adds a test is `fix`; new tests on their own are `test`.

## Branches: `<type>/<topic>`

- `<topic>` is lowercase kebab-case, 2-4 words, naming the change: `feat/storage-eviction`,
  `fix/ring-buffer-wrap`, `docs/architecture`. No issue numbers, dates or your name.
- One branch per logical change. Never commit new work directly to `main`: if you are on
  `main`, create the branch first (`git switch -c <type>/<topic>`).

## Commit messages: `<type>(<scope>): <subject>`

```
fix(edit): keep CRLF endings when re-indenting

The loose matcher joined lines with \n, so editing a CRLF file converted
it to LF. Convert back after the replacement.

Fixes #42
```

- `<type>` as above; usually the same as the branch type.
- `(<scope>)` is optional: the area touched, one lowercase word (`edit`, `cli`, `hooks`).
- `<subject>`: imperative mood ("add", "fix", not "added"), lowercase first letter, no
  trailing period, and the whole first line at most 72 characters.
- Blank line, then a body wrapped at 72 characters saying why the change is needed and
  anything a reviewer should know. Skip it only for trivial changes.
- Breaking change: add `!` after the type or scope (`feat(api)!: ...`) and a
  `BREAKING CHANGE: <what breaks>` footer.
- Revert: `revert: <subject of the reverted commit>` with the body
  `This reverts commit <sha>.`
- Issue references go in a footer: `Fixes #123`, `Refs #456`.

## Steps when committing

1. `git status` and `git diff`: make sure you understand everything that changed.
2. If on `main` (or a branch for a different change), create the right `<type>/<topic>` branch.
3. Stage only the files that belong to this change, by name. No `git add .` / `git add -A`
   without checking; never stage build outputs, caches, scratch scripts or secrets.
4. Split unrelated changes into separate commits, each with its own type.
5. Write the message, then check the branch and the message with this skill's script
   (it is in the skill directory given above):

   ```bash
   bash <skill dir>/scripts/check.sh branch "$(git branch --show-current)"
   printf '%s\n' "$MESSAGE" | bash <skill dir>/scripts/check.sh message
   ```

   Fix everything it reports before committing.
6. Commit, then show `git log -1 --stat`.

Extra context from the user, if any: $ARGUMENTS
