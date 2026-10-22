#!/usr/bin/env bash
#
# scripts/clean-render-artifacts.sh
#
# Remove generated render outputs (Netpbm images and VTK ImageData volumes)
# left behind by the examples and benchmarks. These are the three extensions
# `.gitignore` already classifies as artifacts, and a single wind-tunnel run
# writes close to a gigabyte of them. Pass `--dry-run` to print what would be
# removed without touching disk.
#
# Two safety properties, both deliberate: build trees and the vendored
# reference clones are pruned rather than walked, and any file that git tracks
# is skipped. The second one is what keeps `examples/data/astronaut.pgm` — an
# input the examples read, not an output they write — from ever being deleted,
# without hard-coding its path here.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

dry_run=0
for arg in "$@"; do
  case "${arg}" in
    -n|--dry-run) dry_run=1 ;;
    -h|--help)
      cat <<EOF
Usage: ${0##*/} [--dry-run]

Removes untracked \`*.pgm\`, \`*.ppm\`, and \`*.vti\` files from the working tree.
Tracked files are never removed, and \`build*\`, \`Neon*\`, \`XLB\`, \`third_party\`,
and \`warp\` are not descended into.
EOF
      exit 0
      ;;
    *) echo "${0##*/}: unknown argument '${arg}'" >&2; exit 2 ;;
  esac
done

declare -A tracked=()
while IFS= read -r -d '' path; do
  tracked["${path}"]=1
done < <(git ls-files -z)

removed=0
kept=0
failed=0
total_bytes=0
while IFS= read -r -d '' path; do
  path="${path#./}"
  if [[ -n "${tracked[${path}]:-}" ]]; then
    kept=$((kept + 1))
    continue
  fi
  bytes="$(stat -c '%s' "${path}")"
  human="$(numfmt --to=iec --suffix=B "${bytes}")"
  if [[ ${dry_run} -eq 1 ]]; then
    echo "would remove: ${path} (${human})"
  elif ! rm -f -- "${path}" 2>/dev/null; then
    # Container-written outputs land root-owned under docker-logs/; report them
    # and keep sweeping rather than aborting the run.
    echo "cannot remove (needs sudo): ${path} (${human})" >&2
    failed=$((failed + 1))
    continue
  else
    echo "removed: ${path} (${human})"
  fi
  removed=$((removed + 1))
  total_bytes=$((total_bytes + bytes))
done < <(find . \
  \( -name .git -o -name 'build*' -o -name 'Neon*' -o -name XLB -o -name third_party -o -name warp \) -prune -o \
  -type f \( -name '*.pgm' -o -name '*.ppm' -o -name '*.vti' \) -print0)

if [[ ${removed} -eq 0 && ${failed} -eq 0 ]]; then
  echo "No render artifacts to clean."
elif [[ ${removed} -gt 0 ]]; then
  if [[ ${dry_run} -eq 1 ]]; then
    verb="would free"
  else
    verb="freed"
  fi
  echo "${removed} file(s), ${verb} $(numfmt --to=iec --suffix=B "${total_bytes}")."
fi
if [[ ${kept} -gt 0 ]]; then
  echo "${kept} tracked file(s) skipped."
fi
if [[ ${failed} -gt 0 ]]; then
  echo "${failed} file(s) left in place (insufficient permissions)."
fi
