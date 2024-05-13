import copy
import ctypes
from enum import Enum

import py_neon


class Partition(ctypes.Structure):
    _fields_ = [
        ("mDataView", py_neon.Data_view),
        ("mMem", ctypes.POINTER(ctypes.c_int)),
        ("mDim", py_neon.Index_3d),
        ("mZHaloRadius", ctypes.c_int),
        ("mZBoundaryRadius", ctypes.c_int),
        ("mPitch1", ctypes.c_uint64),
        ("mPitch2", ctypes.c_uint64),
        ("mPitch3", ctypes.c_uint64),
        ("mPitch4", ctypes.c_uint64),
        ("mOrigin", py_neon.Index_3d),
        ("mCardinality", ctypes.c_int),
        ("mFullGridSize", py_neon.Index_3d),
        ("mPeriodicZ", ctypes.c_bool),
        ("mStencil", ctypes.POINTER(ctypes.c_int)),
    ]

    def __str__(self):
        str = "<DSpan: addr=%ld>" % (ctypes.addressof(self))
        str += f"\n\tdataView: {self.mDataView}"
        str += f"\n\tmMem: {self.mMem}"
        str += f"\n\tmDim: {self.mDim}"
        str += f"\n\tmZHaloRadius: {self.mZHaloRadius}"
        str += f"\n\tmZBoundaryRadius: {self.mZBoundaryRadius}"
        str += f"\n\tmPitch1: {self.mPitch1}"
        str += f"\n\tmPitch2: {self.mPitch2}"
        str += f"\n\tmPitch3: {self.mPitch3}"
        str += f"\n\tmPitch4: {self.mPitch4}"
        str += f"\n\tmOrigin: {self.mOrigin}"
        str += f"\n\tmCardinality: {self.mCardinality}"
        str += f"\n\tmFullGridSize: {self.mFullGridSize}"
        str += f"\n\tmPeriodicZ: {self.mPeriodicZ}"
        str += f"\n\tmStencil: {self.mStencil}"
        return str

    @staticmethod
    def fields_():
        return Partition._fields_

