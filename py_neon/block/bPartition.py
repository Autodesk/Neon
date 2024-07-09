import copy
import ctypes
from enum import Enum

import py_neon
from py_neon import Py_neon

class bPartitionInt(ctypes.Structure):
    _fields_ = [
        ("setIdx", ctypes.c_int),
        ("mCardinality", ctypes.c_int),
        ("mMem", ctypes.POINTER(ctypes.c_int)), 
        ("mBlockConnectivity", ctypes.POINTER(ctypes.c_uint32)),
        ("mMask", ctypes.POINTER(ctypes.c_uint32)),  
        ("mOrigin", ctypes.POINTER(py_neon.Index_3d)),
        ("mStencilNghIndex", ctypes.POINTER(ctypes.c_int)), 
        ("mDomainSize", py_neon.Index_3d)
    ]

    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.py_neon.lib.bGrid_bField_bPartition_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.bGrid_bField_bPartition_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 8)()  # Since there are 8 offsets
        self.py_neon.lib.bGrid_bField_bPartition_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]
    
    def __str__(self):
        str_repr = f"<bPartition: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tsetIdx: {self.setIdx} (offset: {bPartitionInt.setIdx.offset})"
        str_repr += f"\n\tmCardinality: {self.mCardinality} (offset: {bPartitionInt.mCardinality.offset})"
        str_repr += f"\n\tmMem: {self.mMem} (offset: {bPartitionInt.mMem.offset})"
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity} (offset: {bPartitionInt.mBlockConnectivity.offset})"
        str_repr += f"\n\tmMask: {self.mMask} (offset: {bPartitionInt.mMask.offset})"
        str_repr += f"\n\tmOrigin: {self.mOrigin} (offset: {bPartitionInt.mOrigin.offset})"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex} (offset: {bPartitionInt.mStencilNghIndex.offset})"
        str_repr += f"\n\tmDomainSize: {self.mDomainSize} (offset: {bPartitionInt.mDomainSize.offset})"
        return str_repr

    def get_offsets(self):
        return [bPartitionInt.setIdx.offset, bPartitionInt.mCardinality.offset, bPartitionInt.mMem.offset, bPartitionInt.mBlockConnectivity.offset, 
                bPartitionInt.mMask.offset, bPartitionInt.mOrigin.offset, bPartitionInt.mStencilNghIndex.offset, bPartitionInt.mDomainSize.offset]
