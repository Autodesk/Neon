import copy
import ctypes
from enum import Enum


from py_neon import Py_neon

class Field(object):
    def __init__(self,
                 py_neon: Py_neon,
                 dgrid_handle: ctypes.c_uint64
                 ):
        self.py_neon = py_neon
        self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
        self.help_load_api()
        self.help_new()

    def __del__(self):
        self.help_delete()

    def help_load_api(self):
        # Importing new functions
        ## new_field
        self.py_neon.lib.dGrid_dField_new.argtypes = [self.handle_type,
                                                      self.handle_type]
        ## delete_field
        self.py_neon.lib.dGrid_dField_delete.argtypes = [self.handle_type]

    def help_new(self):
        if self.handle == 0:
            raise Exception('DGrid: Invalid handle')

        res = self.py_neon.lib.dGrid_dField_new(self.handle)
        if res != 0:
            raise Exception('DGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.py_neon.lib.dGrid_dField_delete(self.handle)
        if res != 0:
            raise Exception('Failed to delete field')

