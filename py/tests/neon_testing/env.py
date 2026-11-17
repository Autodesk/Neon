"""Select which ``neon`` package the tests import.

Order of preference:

1. ``NEON_USE_SOURCE_PY=1``: the source tree (``Neon/py``).
2. An installed ``neon`` that ships ``liblibNeonPy.so`` (wheel install).
3. The source tree as a fallback.

Set ``NEON_TEST_VERBOSE=1`` to print the choice.
"""

import logging
import os
import sys

_SOURCE_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_log = logging.getLogger("neon_testing")


def _report(message: str) -> None:
    if os.environ.get("NEON_TEST_VERBOSE"):
        print(message)
    _log.debug(message)


def update_pythonpath() -> None:
    if os.environ.get("NEON_USE_SOURCE_PY"):
        if _SOURCE_PY not in sys.path:
            sys.path.insert(0, _SOURCE_PY)
        _report(f"Using source neon Python from: {_SOURCE_PY}")
        return

    try:
        import neon
    except ImportError:
        neon = None

    if neon is not None and getattr(neon, "__file__", None):
        neon_dir = os.path.dirname(neon.__file__)
        if os.path.exists(os.path.join(neon_dir, "liblibNeonPy.so")):
            _report(f"Using installed neon from: {neon_dir}")
            return
        for name in [m for m in sys.modules if m == "neon" or m.startswith("neon.")]:
            del sys.modules[name]

    if _SOURCE_PY not in sys.path:
        sys.path.insert(0, _SOURCE_PY)
    _report(f"Using source neon Python (fallback) from: {_SOURCE_PY}")
