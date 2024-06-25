import copy
import ctypes
from enum import Enum

import py_neon


class dPartitionInt(ctypes.Structure):
    _fields_ = [
        ("mDataView", py_neon.DataView),
        ("mDim", py_neon.Index_3d),
        ("mMem", ctypes.POINTER(ctypes.c_int)),
        ("mZHaloRadius", ctypes.c_int),
        ("mZBoundaryRadius", ctypes.c_int),
        ("mPitch1", ctypes.c_uint64),
        ("mPitch2", ctypes.c_uint64),
        ("mPitch3", ctypes.c_uint64),
        ("mPitch4", ctypes.c_uint64),
        ("mPrtID", ctypes.c_int),
        ("mOrigin", py_neon.Index_3d),
        ("mCardinality", ctypes.c_int),
        ("mFullGridSize", py_neon.Index_3d),
        ("mPeriodicZ", ctypes.c_bool),
        ("mStencil", ctypes.POINTER(ctypes.c_int)),
    ]

    def __str__(self):    
        def get_offset(field_name):
                return ctypes.offsetof(dPartitionInt, field_name)

        str = f"<DPartition: addr={ctypes.addressof(self):#x}>"
        str += f"\n\tdataView: {self.mDataView} (offset: {get_offset('mDataView')})"
        str += f"\n\tmDim: {self.mDim} (offset: {get_offset('mDim')})"
        str += f"\n\tmMem: {self.mMem} (offset: {get_offset('mMem')})"
        str += f"\n\tmZHaloRadius: {self.mZHaloRadius} (offset: {get_offset('mZHaloRadius')})"
        str += f"\n\tmZBoundaryRadius: {self.mZBoundaryRadius} (offset: {get_offset('mZBoundaryRadius')})"
        str += f"\n\tmPitch1: {self.mPitch1} (offset: {get_offset('mPitch1')})"
        str += f"\n\tmPitch2: {self.mPitch2} (offset: {get_offset('mPitch2')})"
        str += f"\n\tmPitch3: {self.mPitch3} (offset: {get_offset('mPitch3')})"
        str += f"\n\tmPitch4: {self.mPitch4} (offset: {get_offset('mPitch4')})"
        str += f"\n\tmPrtID: {self.mPrtID} (offset: {get_offset('mPrtID')})"
        str += f"\n\tmOrigin: {self.mOrigin} (offset: {get_offset('mOrigin')})"
        str += f"\n\tmCardinality: {self.mCardinality} (offset: {get_offset('mCardinality')})"
        str += f"\n\tmFullGridSize: {self.mFullGridSize} (offset: {get_offset('mFullGridSize')})"
        str += f"\n\tmPeriodicZ: {self.mPeriodicZ} (offset: {get_offset('mPeriodicZ')})"
        str += f"\n\tmStencil: {self.mStencil} (offset: {get_offset('mStencil')})"
        return str


