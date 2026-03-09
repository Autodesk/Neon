import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

def update_pythonpath():
    """Update PYTHONPATH for development mode only.
    
    If neon is already installed (e.g., from a wheel), don't modify the path.
    This allows tests to run against either the installed package or the source.
    """
    try:
        import neon
        # neon is already importable, check if it has the native library
        if hasattr(neon, '__file__') and neon.__file__:
            neon_dir = os.path.dirname(neon.__file__)
            lib_path = os.path.join(neon_dir, 'liblibNeonPy.so')
            if os.path.exists(lib_path):
                # Installed package with native library - don't modify path
                print(f"Using installed neon from: {neon_dir}")
                logging.debug(f"Using installed neon from: {neon_dir}")
                return
    except ImportError:
        pass
    
    # Development mode: add source directory to path
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    sys.path.insert(0, script_dir+'/../')
    print(f"PYTHONPATH (dev mode): {sys.path}")
    logging.debug(f"PYTHONPATH (dev mode): {sys.path}")