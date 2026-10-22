#!/usr/bin/env python3
"""Environment gate for the ``warp``-labelled test class.

Every other Warp test opens with a chain of ``skipTest`` branches -- library not built,
``warp`` not importable, fork not installed, no CUDA device -- so when the environment is
incomplete they each report a pass without executing the kernel launches, external-type
round-trips, and cross-level addressing they exist to cover. Nothing in ctest's output
distinguishes that from a real run: before 2026-08 the whole suite reported ``20/20
passed`` in 0.65 seconds having compiled nothing, because ctest's interpreter had no
``warp`` on its path.

This gate turns that implicit downgrade into an explicit result:

  - Environment complete: pass.
  - Incomplete and ``CARBON_TEST_STRICT`` unset: pass, but print what is missing and which
    coverage is consequently absent, so a local run still reads honestly.
  - Incomplete and ``CARBON_TEST_STRICT=1``: fail. A lane provisioned to exercise the Warp
    layer sets this, so a missing fork or a hidden GPU turns red instead of silently
    reducing every ``warp`` test to a no-op.

One gate is deliberate, mirroring ``tests/test_multi_gpu_gate.cpp``: teaching ~30 skip
sites to consult the environment would duplicate one policy decision thirty times and
drift the moment a test is added. ``ctest -L warp`` lists the tests it speaks for.
"""

import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strict_mode_requested() -> bool:
    raw = os.environ.get("CARBON_TEST_STRICT")
    if raw is None:
        return False
    return raw not in ("", "0", "false", "off", "OFF", "FALSE")


class CarbonWarpGate(unittest.TestCase):
    def test_warp_environment(self) -> None:
        root = _repo_root()
        missing = []

        lib_path = os.path.join(
            root, "py", "carbon", "lib", f"linux_{os.uname().machine}", "libcarbonpy.so"
        )
        if not os.path.isfile(lib_path):
            missing.append(
                f"libcarbonpy.so not staged at {lib_path} "
                "(build the `carbonpy` target with -DCARBON_ENABLE_PYTHON=ON)"
            )

        sys.path.insert(0, os.path.join(root, "py"))
        wp = None
        try:
            import warp as wp  # noqa: WPS433
        except ImportError:
            missing.append(
                "`import warp` failed. The vendored fork lives at <root>/warp; put it on "
                "PYTHONPATH (ctest does this automatically) or run ./scripts/setup_warp.sh"
            )

        if wp is not None:
            repo = getattr(wp, "__file__", "") or ""
            if not repo:
                missing.append(
                    "`warp` resolved to a namespace package with no __file__ -- almost "
                    "always the repo's own warp/ directory shadowing the real package "
                    "because the process started in the repo root"
                )
            elif "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
                missing.append(
                    f"`warp` at {repo} is not the external-type fork this layer needs; "
                    "run ./scripts/setup_warp.sh"
                )

        if not missing:
            try:
                import carbon  # noqa: WPS433

                gate = carbon.init(enable_warp=True)
                devices = gate.lib.carbonpy_device_count()
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                devices = 0
                missing.append(f"carbon.init(enable_warp=True) raised {exc!r}")
            if devices <= 0:
                missing.append("no CUDA devices visible (carbonpy_device_count() == 0)")

        if not missing:
            return

        detail = "\n".join(f"  - {item}" for item in missing)
        if _strict_mode_requested():
            self.fail(
                "CARBON_TEST_STRICT is set but the Warp environment is incomplete:\n"
                f"{detail}\n"
                "Every test labelled `warp` self-skips in this state and reports a pass it "
                "did not earn, so this run cannot substantiate any claim about the Python "
                "binding layer. Fix the above, or unset CARBON_TEST_STRICT to accept the "
                "reduced coverage."
            )

        print(
            "WARNING: the Warp environment is incomplete:\n"
            f"{detail}\n"
            "Tests labelled `warp` will self-skip and report passes they did not earn -- "
            "this run proves nothing about the Python binding layer. Set "
            "CARBON_TEST_STRICT=1 to make that a failure instead.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    unittest.main()
