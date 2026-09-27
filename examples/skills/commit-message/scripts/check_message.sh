#!/usr/bin/env bash
# Reads a commit message on stdin; prints problems and exits 1 if there are any.
msg=$(cat)
subject=$(printf '%s\n' "$msg" | head -n1)
problems=()
[ ${#subject} -gt 60 ] && problems+=("subject is ${#subject} characters (max 60)")
[[ $subject == *. ]] && problems+=("subject ends with a period")
[[ $subject =~ ^(Added|Fixed|Updated|Changed|Removed)\  ]] && problems+=("use the imperative mood (Add, Fix, ...)")
second=$(printf '%s\n' "$msg" | sed -n 2p)
[ -n "$second" ] && problems+=("the second line must be blank")
while IFS= read -r line; do
  [ ${#line} -gt 72 ] && { problems+=("body line longer than 72 characters"); break; }
done < <(printf '%s\n' "$msg" | tail -n +3)
if [ ${#problems[@]} -gt 0 ]; then printf '%s\n' "${problems[@]}"; exit 1; fi
echo "ok"
