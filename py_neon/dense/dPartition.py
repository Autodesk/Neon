import copy
import ctypes
from enum import Enum

import py_neon
from py_neon import Py_neon


class dPartitionGeneric(ctypes.Structure):
    # _fields_ = [
    #     ("mDataView", py_neon.DataView),
    #     ("mDim", py_neon.Index_3d),
    #     ("mMem", ctypes.POINTER(ctypes.c_int)),
    #     ("mZHaloRadius", ctypes.c_int),
    #     ("mZBoundaryRadius", ctypes.c_int),
    #     ("mPitch1", ctypes.c_uint64),
    #     ("mPitch2", ctypes.c_uint64),
    #     ("mPitch3", ctypes.c_uint64),
    #     ("mPitch4", ctypes.c_uint64),
    #     ("mPrtID", ctypes.c_int),
    #     ("mOrigin", py_neon.Index_3d),
    #     ("mCardinality", ctypes.c_int),
    #     ("mFullGridSize", py_neon.Index_3d),
    #     ("mPeriodicZ", ctypes.c_bool),
    #     ("mStencil", ctypes.POINTER(ctypes.c_int)),
    # ]

    def __init__(self):
        try:
            self.py_neon: Py_neon = Py_neon()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.py_neon.lib.dGrid_dField_dPartition_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.py_neon.lib.dGrid_dField_dPartition_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 15)()  # Since there are 15 offsets
        self.py_neon.lib.dGrid_dField_dPartition_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]

    def __str__(self):
        str = f"<dPartition: addr={ctypes.addressof(self):#x}>"
        str += f"\n\tdataView: {self.mDataView} (offset: {dPartitionInt.mDataView.offset})"
        str += f"\n\tmDim: {self.mDim} (offset: {dPartitionInt.mDim.offset})"
        str += f"\n\tmMem: {self.mMem} (offset: {dPartitionInt.mMem.offset})"
        str += f"\n\tmZHaloRadius: {self.mZHaloRadius} (offset: {dPartitionInt.mZHaloRadius.offset})"
        str += f"\n\tmZBoundaryRadius: {self.mZBoundaryRadius} (offset: {dPartitionInt.mZBoundaryRadius.offset})"
        str += f"\n\tmPitch1: {self.mPitch1} (offset: {dPartitionInt.mPitch1.offset})"
        str += f"\n\tmPitch2: {self.mPitch2} (offset: {dPartitionInt.mPitch2.offset})"
        str += f"\n\tmPitch3: {self.mPitch3} (offset: {dPartitionInt.mPitch3.offset})"
        str += f"\n\tmPitch4: {self.mPitch4} (offset: {dPartitionInt.mPitch4.offset})"
        str += f"\n\tmPrtID: {self.mPrtID} (offset: {dPartitionInt.mPrtID.offset})"
        str += f"\n\tmOrigin: {self.mOrigin} (offset: {dPartitionInt.mOrigin.offset})"
        str += f"\n\tmCardinality: {self.mCardinality} (offset: {dPartitionInt.mCardinality.offset})"
        str += f"\n\tmFullGridSize: {self.mFullGridSize} (offset: {dPartitionInt.mFullGridSize.offset})"
        str += f"\n\tmPeriodicZ: {self.mPeriodicZ} (offset: {dPartitionInt.mPeriodicZ.offset})"
        str += f"\n\tmStencil: {self.mStencil} (offset: {dPartitionInt.mStencil.offset})"
        return str
    
    def get_offsets(self):
        return [dPartitionInt.mDataView.offset, dPartitionInt.mDim.offset, dPartitionInt.mMem.offset, dPartitionInt.mZHaloRadius.offset, 
                dPartitionInt.mZBoundaryRadius.offset, dPartitionInt.mPitch1.offset, dPartitionInt.mPitch2.offset, dPartitionInt.mPitch3.offset, 
                dPartitionInt.mPitch4.offset, dPartitionInt.mPrtID.offset, dPartitionInt.mOrigin.offset, dPartitionInt.mCardinality.offset, 
                dPartitionInt.mFullGridSize.offset, dPartitionInt.mPeriodicZ.offset, dPartitionInt.mStencil.offset]

def factory_dPartition(mem_type):
    """
    Creates a new class based on dPartitionGeneric where the mMem field's type is set to mem_type.

    :param mem_type: The type to be used for the mMem field (e.g., ctypes.POINTER(ctypes.c_double)).
    :return: A new class with the same structure as dPartitionGeneric, but with mMem of type mem_type.
    """

    # Define the _fields_ structure with the dynamic mMem field
    fields = [
        ("mDataView", py_neon.DataView),
        ("mDim", py_neon.Index_3d),
        ("mMem", ctypes.POINTER(mem_type)),
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

    # Create the new class dynamically
    new_class = type(
        f'dPartitionGeneric_{mem_type.__name__}',  # Class name with mem_type name appended
        (ctypes.Structure,),  # Base classes
        {
            '_fields_': fields,
            '__init__': dPartitionGeneric.__init__,
            '_help_load_api': dPartitionGeneric._help_load_api,
            'get_cpp_field_offsets': dPartitionGeneric.get_cpp_field_offsets,
        }
    )

    return new_class

dPartitionInt = factory_dPartition(ctypes.c_int)
dPartitionFloat = factory_dPartition(ctypes.c_float)
dPartitionDouble = factory_dPartition(ctypes.c_double)