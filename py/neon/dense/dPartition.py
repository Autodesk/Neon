import copy
import ctypes
from enum import Enum

import neon
import warp as wp
import neon.dense.dIndex as dIndex
from neon import Index_3d
from neon import Ngh_idx

class dPartitionGeneric(ctypes.Structure):

    def __init__(self):
        try:
            self.neon_gate = neon.Gate()
        except Exception as e:
            self.handle: ctypes.c_uint64 = ctypes.c_uint64(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self._help_load_api()

    def _help_load_api(self):
        self.neon_gate.lib.dGrid_dField_dPartition_get_member_field_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.neon_gate.lib.dGrid_dField_dPartition_get_member_field_offsets.restype = None

    def get_cpp_field_offsets(self):
        length = ctypes.c_size_t()
        offsets = (ctypes.c_size_t * 15)()  # Since there are 15 offsets
        self.neon_gate.lib.dGrid_dField_dPartition_get_member_field_offsets(offsets, ctypes.byref(length))
        return [offsets[i] for i in range(length.value)]

    def __str__(self):
        #str = f"<dPartition: addr={ctypes.addressof(self):#x}>"
        str = f"<dPartition:>"
        str += f"\n\tdataView: {self.mDataView.value} "
        str += f"\n\tmDim: {self.mDim.__str__()}"
        str += f"\n\tmMem: {self.mMem} "
        str += f"\n\tmZHaloRadius: {self.mZHaloRadius}"
        str += f"\n\tmZBoundaryRadius: {self.mZBoundaryRadius}"
        # str += f"\n\tmPitch1: {self.mPitch1}"
        # str += f"\n\tmPitch2: {self.mPitch2}"
        # str += f"\n\tmPitch3: {self.mPitch3}"
        # str += f"\n\tmPitch4: {self.mPitch4}"
        str += f"\n\tmPrtID: {self.mPrtID}"
        # str += f"\n\tmOrigin: {self.mOrigin}"
        str += f"\n\tmCardinality: {self.mCardinality}"
        # str += f"\n\tmFullGridSize: {self.mFullGridSize}"
        # str += f"\n\tmPeriodicZ: {self.mPeriodicZ}"
        # str += f"\n\tmStencil: {self.mStencil}"
        return str
    
    def get_offsets(self):
        return [dPartitionGeneric.mDataView.offset, dPartitionGeneric.mDim.offset, dPartitionGeneric.mMem.offset, dPartitionGeneric.mZHaloRadius.offset, 
                dPartitionGeneric.mZBoundaryRadius.offset, dPartitionGeneric.mPitch1.offset, dPartitionGeneric.mPitch2.offset, dPartitionGeneric.mPitch3.offset, 
                dPartitionGeneric.mPitch4.offset, dPartitionGeneric.mPrtID.offset, dPartitionGeneric.mOrigin.offset, dPartitionGeneric.mCardinality.offset, 
                dPartitionGeneric.mFullGridSize.offset, dPartitionGeneric.mPeriodicZ.offset, dPartitionGeneric.mStencil.offset]

def factory_dPartition(mem_type):
    """
    Creates a new class based on dPartitionGeneric where the mMem field's type is set to mem_type.

    :param mem_type: The type to be used for the mMem field (e.g., ctypes.POINTER(ctypes.c_double)).
    :return: A new class with the same structure as dPartitionGeneric, but with mMem of type mem_type.
    """

    # Define the _fields_ structure with the dynamic mMem field
    mem_type_ctypes=None
    if mem_type == wp.int8:
        mem_type_ctypes = ctypes.c_int8
    elif mem_type == wp.uint8:
        mem_type_ctypes = ctypes.c_uint8
    elif mem_type == wp.bool:
        mem_type_ctypes = ctypes.c_bool
    elif mem_type == wp.int32:
        mem_type_ctypes = ctypes.c_int32
    elif mem_type == wp.uint32:
        mem_type_ctypes = ctypes.c_uint32
    elif mem_type == wp.int64:
        mem_type_ctypes = ctypes.c_int64
    elif mem_type == wp.uint64:
        mem_type_ctypes = ctypes.c_uint64
    elif mem_type == wp.float32:
        mem_type_ctypes = ctypes.c_float
    elif mem_type == wp.float64:
        mem_type_ctypes = ctypes.c_double
    else:
        raise Exception('dPartition: Unsupported data type')


    fields = [
        ("mDataView", neon.DataView),
        ("mDim", neon.Index_3d),
        ("mMem", ctypes.POINTER(mem_type_ctypes)),
        ("mZHaloRadius", ctypes.c_int),
        ("mZBoundaryRadius", ctypes.c_int),
        ("mPitch1", ctypes.c_uint64),
        ("mPitch2", ctypes.c_uint64),
        ("mPitch3", ctypes.c_uint64),
        ("mPitch4", ctypes.c_uint64),
        ("mPrtID", ctypes.c_int),
        ("mOrigin", neon.Index_3d),
        ("mCardinality", ctypes.c_int),
        ("mFullGridSize", neon.Index_3d),
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
            '__str__': dPartitionGeneric.__str__,
        }
    )

    return new_class

dPartition_int8 = factory_dPartition(wp.int8)
dPartition_uint8 = factory_dPartition(wp.uint8)
dPartition_bool = factory_dPartition(wp.bool)

dPartition_int32 = factory_dPartition(wp.int32)
dPartition_uint32 = factory_dPartition(wp.uint32)

dPartition_int64 = factory_dPartition(wp.int64)
dPartition_uint64 = factory_dPartition(wp.uint64)

dPartition_float32 = factory_dPartition(wp.float32)
dPartition_float64 = factory_dPartition(wp.float64)



def register_builtins():
    supported_types = [(dPartition_int8, 'int8', wp.int8),
                       (dPartition_uint8, 'uint8', wp.uint8),

                       (dPartition_int32, 'int32', wp.int32),
                       (dPartition_uint32, 'uint32', wp.uint32),

                       (dPartition_int64, 'int64', wp.int64),
                       (dPartition_uint64, 'uint64', wp.uint64),

                       (dPartition_float32, 'float32', wp.float32),
                       (dPartition_float64, 'float64', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # register type
        wp.types.add_type(Partition, native_name=f"NeonDensePartition_{suffix}", has_binary_ctor=True)

        # print
        wp.context.add_builtin(
            "neon_print_dbg",
            input_types={"p": Partition},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read",
            input_types={"partition": Partition, 'idx': dIndex, "card": int},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition, 'idx': dIndex, "card": int, "value": Type},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_cardinality",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_ngh",
            input_types={"partition": Partition,
                         'idx': dIndex,
                         'ngh_idx': Ngh_idx,
                         "card": wp.int32,
                         "alternative": Type,
                         'is_valid': wp.bool},
            value_type=Type,
            missing_grad=True,
        )
        wp.context.add_builtin(
            "neon_partition_id",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_device_id",
            input_types={"partition": Partition},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_global_idx",
            input_types={"partition": Partition, 'idx': dIndex},
            value_type=Index_3d,
            missing_grad=True,
        )
