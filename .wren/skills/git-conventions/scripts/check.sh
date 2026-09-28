#!/usr/bin/env bash
# Check a branch name or a commit message against the git conventions.
#   check.sh branch <name>
#   check.sh message < message.txt
# Prints problems and exits 1 if there are any; prints "ok" otherwise.
set -u
TYPES="feat|fix|perf|refactor|revert|chore|test|docs"
problems=()

case "${1:-}" in
  branch)
    name="${2:-}"
    if [[ "$name" == "main" || "$name" == "master" ]]; then
      problems+=("on $name: create a <type>/<topic> branch for the change first")
    elif ! [[ "$name" =~ ^($TYPES)/[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
      problems+=("branch '$name' should be <type>/<topic>: type one of ${TYPES//|/, }, topic in lowercase kebab-case")
    else
      words=$(tr '-' '\n' <<< "${name#*/}" | wc -l | tr -d ' ')
      (( words > 4 )) && problems+=("topic has $words words; keep it to 2-4")
    fi
    ;;
  message)
    msg=$(cat)
    subject=$(head -n1 <<< "$msg")
    if ! [[ "$subject" =~ ^($TYPES)(\([a-z0-9-]+\))?!?:\ .+ ]]; then
      problems+=("first line should be '<type>(<scope>): <subject>' with type one of ${TYPES//|/, }")
    else
      text="${subject#*: }"
      [[ "$text" =~ ^[A-Z] ]] && problems+=("start the subject with a lowercase letter")
      [[ "$text" == *. ]] && problems+=("no period at the end of the subject")
      lower=$(tr '[:upper:]' '[:lower:]' <<< "$text")
      [[ "$lower" =~ ^(added|fixed|updated|changed|removed|adds|fixes|updates)\  ]] &&
        problems+=("use the imperative mood (add, fix, update)")
    fi
    (( ${#subject} > 72 )) && problems+=("first line is ${#subject} characters; max 72")
    if [[ $(wc -l <<< "$msg") -gt 1 ]]; then
      [[ -n "$(sed -n 2p <<< "$msg")" ]] && problems+=("the second line must be blank")
      while IFS= read -r line; do
        if (( ${#line} > 72 )) && ! [[ "$line" =~ ^[[:space:]]*(https?://|\`) ]]; then
          problems+=("body line longer than 72 characters: ${line:0:40}..."); break
        fi
      done < <(tail -n +3 <<< "$msg")
    fi
    if [[ "$subject" =~ ^[a-z]+(\([a-z0-9-]+\))?!: ]] && ! grep -q '^BREAKING CHANGE: ' <<< "$msg"; then
      problems+=("'!' marks a breaking change: add a 'BREAKING CHANGE: ...' footer")
    fi
    ;;
  *)
    echo "usage: check.sh branch <name> | check.sh message < file" >&2
    exit 2
    ;;
esac

if (( ${#problems[@]} )); then
  printf '%s\n' "${problems[@]}"
  exit 1
fi
echo ok
