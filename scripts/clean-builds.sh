#!/usr/bin/env bash
#
# scripts/clean-builds.sh
#
# Remove every CMake build tree at the repo root. Recognises both the
# canonical `build-*` form and the legacy `build_*` form left over from
# earlier configurations. Pass `--dry-run` to print what would be removed
# without touching disk.

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

Removes every \`build*\` directory at the repo root (CMake build trees and
\`docker-logs\` are NOT touched — \`docker-logs\` is gitignored separately).
EOF
      exit 0
      ;;
    *) echo "${0##*/}: unknown argument '${arg}'" >&2; exit 2 ;;
  esac
done

shopt -s nullglob
candidates=( build build-* build_* )

if [[ ${#candidates[@]} -eq 0 ]]; then
  echo "No build trees to clean."
  exit 0
fi

for d in "${candidates[@]}"; do
  if [[ ! -d "${d}" ]]; then continue; fi
  if [[ ${dry_run} -eq 1 ]]; then
    echo "would remove: ${d}/"
  else
    echo "removing: ${d}/"
    rm -rf "${d}"
  fi
done
