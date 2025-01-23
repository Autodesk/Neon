import copy
import ctypes
from enum import Enum
import warp as wp
import neon
import neon.block.bIndex

class bPartitionGeneric(ctypes.Structure):

    def __init__(self):
        pass

    def __str__(self):
        str_repr = f"<bPartition: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmCardinality: {self.mCardinality} )"
        str_repr += f"\n\tmMem: {self.mMem})"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex} "
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity} "
        str_repr += f"\n\tmMask: {self.mMask} "
        str_repr += f"\n\tmOrigin: {self.mOrigin} "
        str_repr += f"\n\tmSetIdx: {self.mSetIdx} "
        str_repr += f"\n\tmMultiResDiscreteIdxSpacing: {self.mMultiResDiscreteIdxSpacing} "
        str_repr += f"\n\tmDomainSize: {self.mDomainSize} "
        return str_repr

    def _help_load_api(self):
        self.neon_gate:neon.Gate =  neon.Gate()


def factory_bPartition(dtype):
    """
    Creates a new class based on bPartitionGeneric where the mMem field's type is set to dtype.

    :param dtype: The type to be used for the mMem field (e.g., ctypes.POINTER(ctypes.c_double)).
    :return: A new class with the same structure as bPartitionGeneric, but with mMem of type dtype.
    """
    neon_gate: neon.Gate = neon.Gate()
    type_mapping = neon_gate.get_type_mapping(dtype)

    fields = [
        ("mCardinality", ctypes.c_int),
        ("mMem", ctypes.POINTER(ctypes.c_int)),
        ("mStencilNghIndex", ctypes.POINTER(ctypes.c_int)),
        ("mBlockConnectivity", ctypes.POINTER(ctypes.c_uint32)),
        ("mMask", ctypes.POINTER(ctypes.c_uint32)),
        ("mOrigin", ctypes.POINTER(neon.Index_3d)),
        ("mSetIdx", ctypes.c_int),
        ("mMultiResDiscreteIdxSpacing", ctypes.c_int),
        ("mDomainSize", neon.Index_3d)

    ]

    # Create the new class dynamically
    suffix = type_mapping['suffix']
    new_class = type(
        f'bPartitionGeneric{suffix}',  # Class name with mem_type name appended
        (ctypes.Structure,),  # Base classes
        {
            '_fields_': fields,
            '__init__': bPartitionGeneric.__init__,
            '_help_load_api': bPartitionGeneric._help_load_api,
            '__str__': bPartitionGeneric.__str__,
        }
    )

    return new_class


bPartition_int8 = factory_bPartition(wp.int8)
bPartition_uint8 = factory_bPartition(wp.uint8)
bPartition_bool = factory_bPartition(wp.bool)

bPartition_int32 = factory_bPartition(wp.int32)
bPartition_uint32 = factory_bPartition(wp.uint32)

bPartition_int64 = factory_bPartition(wp.int64)
bPartition_uint64 = factory_bPartition(wp.uint64)

bPartition_float32 = factory_bPartition(wp.float32)
bPartition_float64 = factory_bPartition(wp.float64)


def register_builtins():
    supported_types = [(bPartition_int8, 'int8', wp.int8),
                       (bPartition_uint8, 'uint8', wp.uint8),

                       (bPartition_int32, 'int32', wp.int32),
                       (bPartition_uint32, 'uint32', wp.uint32),

                       (bPartition_int64, 'int64', wp.int64),
                       (bPartition_uint64, 'uint64', wp.uint64),

                       (bPartition_float32, 'float32', wp.float32),
                       (bPartition_float64, 'float64', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # register type
        wp.types.add_type(Partition, native_name=f"NeonBlockPartition_{suffix}", has_binary_ctor=True)

        # # print
        # wp.context.add_builtin(
        #     "neon_print_dbg",
        #     input_types={"p": Partition},
        #     value_type=None,
        #     missing_grad=True,
        # )

        wp.context.add_builtin(
            "neon_read",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         "card": int},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         "card": int,
                         "value": Type},
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
            "neon_ngh_data",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx,
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
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex},
            value_type=neon.Index_3d,
            missing_grad=True,
        )
