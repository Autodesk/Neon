import copy
import ctypes
from enum import Enum
from py_neon import Py_neon

from py_neon.py_ne import Py_neon as NePy_neon




class allocationCounter(object):
    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        # Importing new functions
        self.py_neon.lib.get_allocation_counter.argtypes = []
        self.py_neon.lib.get_allocation_counter.restype = ctypes.c_int

    def get_allocation_count(self):
        return self.py_neon.lib.get_allocation_counter()
