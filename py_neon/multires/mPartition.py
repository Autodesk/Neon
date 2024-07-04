import copy
import ctypes
from enum import Enum

import py_neon


class mPartitionInt(ctypes.Structure):
    _fields_ = [
        ("mLevel", ctypes.c_int),
        ("mMemParent", ctypes.POINTER(ctypes.c_int)),
        ("mMemChild", ctypes.POINTER(ctypes.c_int)),
        ("mParentBlockID", ctypes.POINTER(ctypes.c_uint32)),
        ("mMaskLowerLevel", ctypes.POINTER(ctypes.c_uint32)),
        ("mMaskUpperLevel", ctypes.POINTER(ctypes.c_uint32)),
        ("mChildBlockID", ctypes.POINTER(ctypes.c_uint32)),
        ("mParentNeighbourBlocks", ctypes.POINTER(ctypes.c_uint32)),
        ("mRefFactors", ctypes.POINTER(ctypes.c_int)),
        ("mSpacing", ctypes.POINTER(ctypes.c_int))
    ]

    def __str__(self):
        def get_offset(field_name):
            return ctypes.offsetof(mPartitionInt, field_name)

        str_repr = f"<mPartitionInt: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmLevel: {self.mLevel} (offset: {get_offset('mLevel')})"
        str_repr += f"\n\tmMemParent: {self.mMemParent} (offset: {get_offset('mMemParent')})"
        str_repr += f"\n\tmMemChild: {self.mMemChild} (offset: {get_offset('mMemChild')})"
        str_repr += f"\n\tmParentBlockID: {self.mParentBlockID} (offset: {get_offset('mParentBlockID')})"
        str_repr += f"\n\tmMaskLowerLevel: {self.mMaskLowerLevel} (offset: {get_offset('mMaskLowerLevel')})"
        str_repr += f"\n\tmMaskUpperLevel: {self.mMaskUpperLevel} (offset: {get_offset('mMaskUpperLevel')})"
        str_repr += f"\n\tmChildBlockID: {self.mChildBlockID} (offset: {get_offset('mChildBlockID')})"
        str_repr += f"\n\tmParentNeighbourBlocks: {self.mParentNeighbourBlocks} (offset: {get_offset('mParentNeighbourBlocks')})"
        str_repr += f"\n\tmRefFactors: {self.mRefFactors} (offset: {get_offset('mRefFactors')})"
        str_repr += f"\n\tmSpacing: {self.mSpacing} (offset: {get_offset('mSpacing')})"
        return str_repr