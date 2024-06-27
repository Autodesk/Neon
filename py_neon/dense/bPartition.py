import copy
import ctypes
from enum import Enum

import py_neon


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

    def __str__(self):
        def get_offset(field_name):
            return ctypes.offsetof(bPartitionInt, field_name)

        str_repr = f"<bPartition: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tsetIdx: {self.setIdx} (offset: {get_offset('setIdx')})"
        str_repr += f"\n\tmCardinality: {self.mCardinality} (offset: {get_offset('mCardinality')})"
        str_repr += f"\n\tmMem: {self.mMem} (offset: {get_offset('mMem')})"
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity} (offset: {get_offset('mBlockConnectivity')})"
        str_repr += f"\n\tmMask: {self.mMask} (offset: {get_offset('mMask')})"
        str_repr += f"\n\tmOrigin: {self.mOrigin} (offset: {get_offset('mOrigin')})"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex} (offset: {get_offset('mStencilNghIndex')})"
        str_repr += f"\n\tmDomainSize: {self.mDomainSize} (offset: {get_offset('mDomainSize')})"
        return str_repr

