#!/usr/bin/env bash
# Run the Neon Python test suite.
#
# Each test_*.py is run in its OWN process. Warp + Neon initialize global GPU
# state (CUDA contexts via cuCtxCreate_v4, kernel cache, etc.) that does not
# survive being re-initialized many times in a single interpreter, so running
# everything in one `unittest discover` process segfaults. Per-process
# isolation is the reliable way to run them as a suite.
#
# Usage (from an environment with the `neon` deps and CUDA on PATH):
#   ./run_tests.sh
# Honors $PYTHON (default: python3) so it works under `conda run -n neon` too.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Set PYTHONPATH / LD_LIBRARY_PATH for the local Neon build.
# env.sh derives these from $PWD, so it must be sourced from the repo root.
# (It also references LD_LIBRARY_PATH unguarded, so seed it for `set -u`.)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "./env.sh" export >/dev/null

cd "${SCRIPT_DIR}"

pass=0
fail=0
failed_modules=()

for f in test_*.py; do
    mod="${f%.py}"
    printf '=== %-32s ' "${mod}"
    # shellcheck disable=SC2086  # $PYTHON may be a multi-word command (e.g. "conda run -n neon python3")
    if ${PYTHON} -m unittest "${mod}" >"/tmp/neon_suite_${mod}.log" 2>&1; then
        echo "PASS"
        pass=$((pass + 1))
    else
        echo "FAIL  (see /tmp/neon_suite_${mod}.log)"
        fail=$((fail + 1))
        failed_modules+=("${mod}")
    fi
done

echo
echo "==================================================="
echo "Suite result: ${pass} passed, ${fail} failed"
if [ "${fail}" -ne 0 ]; then
    printf '  failed: %s\n' "${failed_modules[*]}"
    exit 1
fi
