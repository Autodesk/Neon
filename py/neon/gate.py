import ctypes
import os
import warp as wp
import platform

class Gate(object):
    def __init__(self):
        self.handle_type = ctypes.c_void_p
        # get the path of this python file
        current_file_path = os.path.abspath(__file__)
        # get the directory containing the script
        # Determine platform and architecture
        current_platform = platform.system().lower()
        machine = platform.machine().lower()
        
        if current_platform.startswith("linux"):
            platform_key = f"linux_{machine}"
            lib_extension = ".so"
        elif current_platform.startswith("darwin"):
            platform_key = f"macos_{machine}"
            lib_extension = ".dylib" 
        elif current_platform.startswith("windows"):
            platform_key = f"windows_{machine}"
            lib_extension = ".dll"
        else:
            raise RuntimeError(f"Unsupported platform: {current_platform}")
            
        lib_path = os.path.join(os.path.dirname(current_file_path), "lib", platform_key, f"liblibNeonPy{lib_extension}")

        try:
            self.lib =    ctypes.CDLL(lib_path)
        except Exception as e:
            print(f"Failed to load library: {lib_path}")
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


    def get_type_mapping(self, warp_type):
        # returns the corresponding ctypes type
        ret = {}
        try:
            ret['suffix']= self._get_suffix(warp_type)
            ret['ctype']= self.to_ctypes[ret['suffix']]
            ret['warp'] = warp_type
            return ret
        except Exception as e:
            raise Exception(f"Unsupported warp type. {warp_type}: {str(e)}")


    def _get_supported_wp_types(self):
        # returns all values from the to_warp_types dictionary
        return list(self.to_warp_types.values())

    def _get_suffix(self, wpType):
        return self.warp_type_to_string[wpType]
