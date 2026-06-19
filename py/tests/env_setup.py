import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)


def update_pythonpath():
    """Prefer an installed neon wheel; otherwise use the source tree."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_py = os.path.join(script_dir, "..")

    if os.environ.get("NEON_USE_SOURCE_PY"):
        sys.path.insert(0, source_py)
        if os.environ.get("NEON_TEST_VERBOSE"):
            print(f"Using source neon Python from: {source_py}")
        logging.debug("Using source neon Python from: %s", source_py)
        return

    try:
        import neon

        if hasattr(neon, "__file__") and neon.__file__:
            neon_dir = os.path.dirname(neon.__file__)
            lib_path = os.path.join(neon_dir, "liblibNeonPy.so")
            if os.path.exists(lib_path):
                if os.environ.get("NEON_TEST_VERBOSE"):
                    print(f"Using installed neon from: {neon_dir}")
                logging.debug("Using installed neon from: %s", neon_dir)
                return
    except ImportError:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(script_dir, ".."))
    if os.environ.get("NEON_TEST_VERBOSE"):
        print(f"PYTHONPATH (dev mode): {sys.path}")
    logging.debug("PYTHONPATH (dev mode): %s", sys.path)
