#!/usr/bin/env python3
"""Smoke test for libcarbonpy ctypes loading. Skips when the shared library is absent."""

import os
import sys
import unittest


class CarbonPySmoke(unittest.TestCase):
    def test_init(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_dir = os.path.join(root, "py", "carbon", "lib", f"linux_{os.uname().machine}")
        lib_path = os.path.join(lib_dir, "libcarbonpy.so")
        if not os.path.isfile(lib_path):
            self.skipTest(f"libcarbonpy not built at {lib_path}")
        sys.path.insert(0, os.path.join(root, "py"))
        import carbon  # noqa: WPS433

        gate = carbon.init()
        self.assertGreaterEqual(gate.lib.carbonpy_device_count(), 0)


if __name__ == "__main__":
    unittest.main()
