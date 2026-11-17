#!/usr/bin/env bash
# Run the Neon Python test suite.
#
# Layout: one package per area (smoke/, dense/, block/, mres/, lifecycle/),
# shared helpers in neon_testing/. Tests subclass neon_testing.NeonTestCase,
# which initializes Warp/Neon once per process and releases Neon objects
# deterministically, so the whole suite runs in a single interpreter.
#
# Usage (from an environment with the `neon` deps and CUDA on PATH):
#   ./run_tests.sh                # one `unittest discover` process
#   ./run_tests.sh --isolated     # one process per test module
#   ./run_tests.sh mres lifecycle # restrict to some areas
#
# Honors $PYTHON (default: python3), e.g. PYTHON="conda run -n neon python3".
# Set NEON_RUN_TESTS_NO_ENV=1 to skip sourcing ../../env.sh (wheel installs).
# Useful test knobs: NEON_TEST_WARP_MODE=release, NEON_TEST_VERBOSE=1,
# NEON_EXPORT_VTI=1 with NEON_TEST_ARTIFACTS=<dir>.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${NEON_TEST_LOG_DIR:-${TMPDIR:-/tmp}/neon_tests}"

isolated=0
areas=()
for arg in "$@"; do
    case "${arg}" in
        --isolated) isolated=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) areas+=("${arg%/}") ;;
    esac
done
if [ "${#areas[@]}" -eq 0 ]; then
    areas=(smoke dense block mres lifecycle)
fi

if [ -z "${NEON_RUN_TESTS_NO_ENV:-}" ] && [ -f "${REPO_ROOT}/env.sh" ]; then
    # env.sh derives paths from $PWD and reads LD_LIBRARY_PATH unguarded.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    cd "${REPO_ROOT}"
    # shellcheck disable=SC1091
    source "./env.sh" export >/dev/null
fi

cd "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

if [ "${isolated}" -eq 0 ]; then
    status=0
    for area in "${areas[@]}"; do
        echo "=== ${area}"
        # shellcheck disable=SC2086  # $PYTHON may be a multi-word command
        ${PYTHON} -m unittest discover -s "${area}" -t . -p 'test_*.py' || status=1
    done
    exit "${status}"
fi

pass=0
fail=0
failed_modules=()
for area in "${areas[@]}"; do
    for f in "${area}"/test_*.py; do
        [ -e "${f}" ] || continue
        mod="${f%.py}"
        mod="${mod//\//.}"
        log="${LOG_DIR}/${mod}.log"
        printf '=== %-40s ' "${mod}"
        # shellcheck disable=SC2086
        if ${PYTHON} -m unittest "${mod}" >"${log}" 2>&1; then
            echo "PASS"
            pass=$((pass + 1))
        else
            echo "FAIL  (see ${log})"
            fail=$((fail + 1))
            failed_modules+=("${mod}")
        fi
    done
done

echo
echo "==================================================="
echo "Suite result: ${pass} passed, ${fail} failed"
if [ "${fail}" -ne 0 ]; then
    printf '  failed: %s\n' "${failed_modules[*]}"
    exit 1
fi
