import copy
import ctypes
from enum import Enum
from py_neon import Py_neon

from py_neon import DataView

class bSpan(ctypes.Structure):
    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()

    def help_load_api(self):

        # grid_new
        self.py_neon.lib.bGrid_bSpan_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.bGrid_bSpan_get_member_field_offsets.restype = ctypes.POINTER(ctypes.c_size_t)

    def get_member_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = self.py_neon.lib.bGrid_bSpan_get_member_field_offsets(ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]

    _fields_ = [
        ("mFirstDataBlockOffset", ctypes.c_uint32),
        ("mActiveMask", ctypes.POINTER(ctypes.c_uint32)),
        ("mDataView", DataView)
    ]

    def __str__(self):
        def get_offset(field_name):
            return ctypes.offsetof(bSpan, field_name)

        str_repr = f"<bSpan: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmFirstDataBlockOffset: {self.mFirstDataBlockOffset} (offset: {get_offset('mFirstDataBlockOffset')})"
        str_repr += f"\n\tmActiveMask: {self.mActiveMask} (offset: {get_offset('mActiveMask')})"
        str_repr += f"\n\tmDataView: {self.mDataView} (offset: {get_offset('mDataView')})"
        return str_repr
    
    def get_offsets(self):
        def get_offset(field_name):
            return ctypes.offsetof(bSpan, field_name)
        return [get_offset('mFirstDataBlockOffset'), get_offset('mActiveMask'), get_offset('mDataView')]

    @staticmethod
    def fields_():
        return bSpan._fields_