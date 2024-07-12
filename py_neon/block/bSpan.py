import copy
import ctypes
from enum import Enum
from py_neon import Py_neon

from py_neon import DataView

class bSpan(ctypes.Structure):
    _fields_ = [
        ("vtablePtr", ctypes.c_uint64),
        ("mFirstDataBlockOffset", ctypes.c_uint32),
        ("mActiveMask", ctypes.POINTER(ctypes.c_uint32)),
        ("mDataView", DataView)
    ]

    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.py_neon.lib.bGrid_bSpan_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.bGrid_bSpan_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 3)()  # Assuming there are 3 offsets
        self.py_neon.lib.bGrid_bSpan_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]


    def __str__(self):
        str_repr = f"<bSpan: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmFirstDataBlockOffset: {self.mFirstDataBlockOffset} (offset: {bSpan.mFirstDataBlockOffset.offset})"
        str_repr += f"\n\tmActiveMask: {self.mActiveMask} (offset: {bSpan.mActiveMask.offset})"
        str_repr += f"\n\tmDataView: {self.mDataView} (offset: {bSpan.mDataView.offset})"
        return str_repr
    
    def get_offsets(self):
        return [bSpan.mFirstDataBlockOffset.offset, bSpan.mActiveMask.offset, bSpan.mDataView.offset]

    @staticmethod
    def fields_():
        return bSpan._fields_