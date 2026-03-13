"""
Gate Module for Neon Computing Framework

This module provides the Gate class and library loading utilities that serve
as the bridge between Python and the C++ Neon library.
"""

import ctypes
import os
import sys
import warp as wp


def _find_neon_library() -> str:
    """
    Find the libNeonPy shared library.
    
    Searches for the native library in several locations:
    1. Environment variable NEON_LIB_PATH (if set)
    2. Same directory as this Python module (for wheel installs)
    3. Build directories (for development)
    
    Returns:
        str: Absolute path to the library file.
    
    Raises:
        FileNotFoundError: If the library cannot be found in any location.
    
    Note:
        The library name varies by platform:
        - Linux: liblibNeonPy.so
        - macOS: liblibNeonPy.dylib
        - Windows: liblibNeonPy.dll
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if sys.platform == "win32":
        lib_name = "liblibNeonPy.dll"
    elif sys.platform == "darwin":
        lib_name = "liblibNeonPy.dylib"
    else:
        lib_name = "liblibNeonPy.so"
    
    search_paths = [
        os.path.join(current_dir, lib_name),
        os.path.join(current_dir, "..", "..", "build", "libNeonPy", lib_name),
        os.path.join(current_dir, "..", "..", "build2", "libNeonPy", lib_name),
    ]
    
    env_path = os.environ.get("NEON_LIB_PATH")
    if env_path:
        search_paths.insert(0, os.path.join(env_path, lib_name))
        search_paths.insert(0, env_path)
    
    for path in search_paths:
        if os.path.isfile(path):
            return path
    
    raise FileNotFoundError(
        f"Could not find {lib_name}. Searched paths:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\n\nSet NEON_LIB_PATH environment variable to the directory containing the library."
    )


class Gate(object):
    """
    Interface to the Neon C++ library via ctypes.
    
    The Gate class loads the libNeonPy shared library and provides type
    mappings between Python/Warp types and C++ types. It serves as the
    foundation for all Python-to-C++ communication in Neon.
    
    Attributes:
        handle_type: The ctypes type used for C++ object handles (c_void_p).
        lib: The loaded ctypes CDLL library object.
        to_warp_types (dict): Mapping from type name strings to Warp types.
        warp_type_to_string (dict): Mapping from Warp types to type name strings.
        warp_type_to_cpp_type_string (dict): Mapping from Warp types to C++ type strings.
        to_ctypes (dict): Mapping from type name strings to ctypes types.
    
    Example:
        >>> gate = neon.Gate()
        >>> gate.lib.backend_new(...)  # Call C++ function
        >>> 
        >>> # Get type mappings
        >>> mapping = gate.get_type_mapping(wp.float32)
        >>> print(mapping['ctype'])  # ctypes.c_float
    
    Note:
        Most users don't need to interact with Gate directly. It is used
        internally by Backend, Grid, Field, and other Neon classes.
    """
    
    def __init__(self):
        """
        Load the Neon library and initialize type mappings.
        
        Raises:
            FileNotFoundError: If the Neon library cannot be found.
            OSError: If the library fails to load (e.g., missing dependencies).
        """
        self.handle_type = ctypes.c_void_p
        lib_path = _find_neon_library()

        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            from .logging import logger
            logger.error(f"Failed to load library: {lib_path}")
            raise e

        self.to_warp_types = {
            "bool": wp.bool,
            "int8": wp.int8,
            "uint8": wp.uint8,
            # "int16": wp.int16,
            # "uint16": wp.uint16,
            "int32": wp.int32,
            "uint32": wp.uint32,
            "int64": wp.int64,
            "uint64": wp.uint64,
            # "float16": wp.float16,
            "float32": wp.float32,  # alias: float
            "float64": wp.float64,  # alias: double
            "float": wp.float32,  # alias for float32
            "int": wp.int32,  # alias for int32
        }

        self.warp_type_to_string = {
            wp.bool: "bool",
            wp.int8: "int8",
            wp.uint8: "uint8",
            wp.int16: "int16",
            wp.uint16: "uint16",
            wp.int32: "int32",
            wp.uint32: "uint32",
            wp.int64: "int64",
            wp.uint64: "uint64",
            wp.float16: "float16",
            wp.float32: "float32",
            wp.float64: "float64",
        }

        self.warp_type_to_cpp_type_string = {
            wp.bool: "bool",
            wp.int8: "int8_t",
            wp.uint8: "uint8_t",
            wp.int32: "int32_t",
            wp.uint32: "uint32_t",
            wp.int64: "int64_t",
            wp.uint64: "uint64_t",
            wp.float32: "float",
            wp.float64: "double",
        }
        # Dictionary mapping basic scalar types to ctypes types
        self.to_ctypes = {
            "bool": ctypes.c_bool,
            "int8": ctypes.c_int8,
            "uint8": ctypes.c_uint8,
            "int16": ctypes.c_int16,
            "uint16": ctypes.c_uint16,
            "int32": ctypes.c_int32,
            "uint32": ctypes.c_uint32,
            "int64": ctypes.c_int64,
            "uint64": ctypes.c_uint64,
            "float16": None,  # ctypes has no built-in half-precision float type
            "float32": ctypes.c_float,  # single-precision
            "float64": ctypes.c_double,  # double-precision
            "float": ctypes.c_float,  # alias for float32
            "int": ctypes.c_int32,  # alias for int32
        }


    def get_type_mapping(self, warp_type) -> dict:
        """
        Get type mapping information for a Warp type.
        
        Returns a dictionary containing type information needed for C++ interop.
        
        Args:
            warp_type: A Warp type (e.g., wp.float32, wp.int32).
        
        Returns:
            dict: Type mapping with keys:
                - 'suffix': Type name string (e.g., 'float32')
                - 'ctype': Corresponding ctypes type
                - 'warp': The original Warp type
        
        Raises:
            Exception: If the Warp type is not supported.
        
        Example:
            >>> gate = neon.Gate()
            >>> mapping = gate.get_type_mapping(wp.float32)
            >>> print(mapping['suffix'])  # 'float32'
            >>> print(mapping['ctype'])   # <class 'ctypes.c_float'>
        """
        ret = {}
        try:
            ret['suffix'] = self._get_suffix(warp_type)
            ret['ctype'] = self.to_ctypes[ret['suffix']]
            ret['warp'] = warp_type
            return ret
        except Exception as e:
            raise Exception(f"Unsupported warp type. {warp_type}: {str(e)}")

    def _get_supported_wp_types(self) -> list:
        """Internal: Get list of all supported Warp types."""
        return list(self.to_warp_types.values())

    def _get_suffix(self, wpType) -> str:
        """Internal: Get type name suffix for a Warp type."""
        return self.warp_type_to_string[wpType]
