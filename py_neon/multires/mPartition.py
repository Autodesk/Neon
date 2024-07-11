import copy
import ctypes
from enum import Enum

import py_neon
from py_neon import Py_neon


class mPartitionInt(ctypes.Structure):
    _fields_ = [
        ("mCardinality", ctypes.c_int),
        ("mMem", ctypes.POINTER(ctypes.c_int)), 
        ("mStencilNghIndex", ctypes.POINTER(ctypes.c_int)), 
        ("mBlockConnectivity", ctypes.POINTER(ctypes.c_uint32)),
        ("mMask", ctypes.POINTER(ctypes.c_uint32)),  
        ("mOrigin", ctypes.POINTER(py_neon.Index_3d)),
        ("mSetIdx", ctypes.c_int),
        ("mMultiResDiscreteIdxSpacing", ctypes.c_int),
        ("mDomainSize", py_neon.Index_3d),
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

    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.py_neon.lib.mGrid_mField_mPartition_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.mGrid_mField_mPartition_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 19)()  # Since there are 19 offsets
        self.py_neon.lib.mGrid_mField_mPartition_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]

    def __str__(self):
        str_repr = f"<mPartitionInt: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmCardinality: {self.mCardinality} (offset: {mPartitionInt.mCardinality.offset})"
        str_repr += f"\n\tmMem: {self.mMem} (offset: {mPartitionInt.mMem.offset})"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex} (offset: {mPartitionInt.mStencilNghIndex.offset})"
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity} (offset: {mPartitionInt.mBlockConnectivity.offset})"
        str_repr += f"\n\tmMask: {self.mMask} (offset: {mPartitionInt.mMask.offset})"
        str_repr += f"\n\tmOrigin: {self.mOrigin} (offset: {mPartitionInt.mOrigin.offset})"
        str_repr += f"\n\tmSetIdx: {self.mSetIdx} (offset: {mPartitionInt.mSetIdx.offset})"
        str_repr += f"\n\tmMultiResDiscreteIdxSpacing: {self.mMultiResDiscreteIdxSpacing} (offset: {mPartitionInt.mMultiResDiscreteIdxSpacing.offset})"
        str_repr += f"\n\tmDomainSize: {self.mDomainSize} (offset: {mPartitionInt.mDomainSize.offset})"
        str_repr += f"\n\tmLevel: {self.mLevel} (offset: {mPartitionInt.mLevel.offset})"
        str_repr += f"\n\tmMemParent: {self.mMemParent} (offset: {mPartitionInt.mMemParent.offset})"
        str_repr += f"\n\tmMemChild: {self.mMemChild} (offset: {mPartitionInt.mMemChild.offset})"
        str_repr += f"\n\tmParentBlockID: {self.mParentBlockID} (offset: {mPartitionInt.mParentBlockID.offset})"
        str_repr += f"\n\tmMaskLowerLevel: {self.mMaskLowerLevel} (offset: {mPartitionInt.mMaskLowerLevel.offset})"
        str_repr += f"\n\tmMaskUpperLevel: {self.mMaskUpperLevel} (offset: {mPartitionInt.mMaskUpperLevel.offset})"
        str_repr += f"\n\tmChildBlockID: {self.mChildBlockID} (offset: {mPartitionInt.mChildBlockID.offset})"
        str_repr += f"\n\tmParentNeighbourBlocks: {self.mParentNeighbourBlocks} (offset: {mPartitionInt.mParentNeighbourBlocks.offset})"
        str_repr += f"\n\tmRefFactors: {self.mRefFactors} (offset: {mPartitionInt.mRefFactors.offset})"
        str_repr += f"\n\tmSpacing: {self.mSpacing} (offset: {mPartitionInt.mSpacing.offset})"
        return str_repr
    
    def get_offsets(self):
        return [mPartitionInt.mCardinality.offset, mPartitionInt.mMem.offset, mPartitionInt.mStencilNghIndex.offset, 
                mPartitionInt.mBlockConnectivity.offset, mPartitionInt.mMask.offset, mPartitionInt.mOrigin.offset, 
                mPartitionInt.mSetIdx.offset, mPartitionInt.mMultiResDiscreteIdxSpacing.offset, mPartitionInt.mDomainSize.offset, 
                mPartitionInt.mLevel.offset, mPartitionInt.mMemParent.offset, mPartitionInt.mMemChild.offset, mPartitionInt.mParentBlockID.offset,
                mPartitionInt.mMaskLowerLevel.offset, mPartitionInt.mMaskUpperLevel.offset, mPartitionInt.mChildBlockID.offset, 
                mPartitionInt.mParentNeighbourBlocks.offset, mPartitionInt.mRefFactors.offset, mPartitionInt.mSpacing.offset]
