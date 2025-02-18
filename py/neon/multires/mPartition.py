import copy
import ctypes
from enum import Enum

import neon
import warp as wp
class mPartitionGeneric(ctypes.Structure):

    def __init__(self):
        self._help_load_api()
        pass

    def __str__(self):
        str_repr = f"<mPartitionInt: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmCardinality: {self.mCardinality} (offset: {self.mPartitionInt.mCardinality.offset})"
        str_repr += f"\n\tmMem: {self.mMem} (offset: {self.mPartitionInt.mMem.offset})"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex} (offset: {self.mPartitionInt.mStencilNghIndex.offset})"
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity} (offset: {self.mPartitionInt.mBlockConnectivity.offset})"
        str_repr += f"\n\tmMask: {self.mMask} (offset: {self.mPartitionInt.mMask.offset})"
        str_repr += f"\n\tmOrigin: {self.mOrigin} (offset: {self.mPartitionInt.mOrigin.offset})"
        str_repr += f"\n\tmSetIdx: {self.mSetIdx} (offset: {self.mPartitionInt.mSetIdx.offset})"
        str_repr += f"\n\tmMultiResDiscreteIdxSpacing: {self.mMultiResDiscreteIdxSpacing} (offset: {self.mPartitionInt.mMultiResDiscreteIdxSpacing.offset})"
        str_repr += f"\n\tmDomainSize: {self.mDomainSize} (offset: {self.mPartitionInt.mDomainSize.offset})"
        str_repr += f"\n\tmLevel: {self.mLevel} (offset: {self.mPartitionInt.mLevel.offset})"
        str_repr += f"\n\tmMemParent: {self.mMemParent} (offset: {self.mPartitionInt.mMemParent.offset})"
        str_repr += f"\n\tmMemChild: {self.mMemChild} (offset: {self.mPartitionInt.mMemChild.offset})"
        str_repr += f"\n\tmParentBlockID: {self.mParentBlockID} (offset: {self.mPartitionInt.mParentBlockID.offset})"
        str_repr += f"\n\tmMaskLowerLevel: {self.mMaskLowerLevel} (offset: {self.mPartitionInt.mMaskLowerLevel.offset})"
        str_repr += f"\n\tmMaskUpperLevel: {self.mMaskUpperLevel} (offset: {self.mPartitionInt.mMaskUpperLevel.offset})"
        str_repr += f"\n\tmChildBlockID: {self.mChildBlockID} (offset: {self.mPartitionInt.mChildBlockID.offset})"
        str_repr += f"\n\tmParentNeighbourBlocks: {self.mParentNeighbourBlocks} (offset: {self.mPartitionInt.mParentNeighbourBlocks.offset})"
        str_repr += f"\n\tmRefFactors: {self.mRefFactors} (offset: {self.mPartitionInt.mRefFactors.offset})"
        str_repr += f"\n\tmSpacing: {self.mSpacing} (offset: {self.mPartitionInt.mSpacing.offset})"
        return str_repr

    def _help_load_api(self):
        self.neon_gate:neon.Gate =  neon.Gate()
        
def factory_mPartition(dtype):
    """
    Creates a new class based on bPartitionGeneric where the mMem field's type is set to dtype.

    :param dtype: The type to be used for the mMem field (e.g., ctypes.POINTER(ctypes.c_double)).
    :return: A new class with the same structure as bPartitionGeneric, but with mMem of type dtype.
    """
    neon_gate: neon.Gate = neon.Gate()
    type_mapping = neon_gate.get_type_mapping(dtype)
    
    bPartition_fields_  = [
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
    
    mPartition_fields_ = [("mLevel", ctypes.c_int),
        ("mMemParent", ctypes.POINTER(ctypes.c_int)),
        ("mMemChild", ctypes.POINTER(ctypes.c_int)),
        ("mParentBlockID", ctypes.POINTER(ctypes.c_uint32)),
        ("mMaskLowerLevel", ctypes.POINTER(ctypes.c_uint32)),
        ("mMaskUpperLevel", ctypes.POINTER(ctypes.c_uint32)),
        ("mChildBlockID", ctypes.POINTER(ctypes.c_uint32)),
        ("mParentNeighbourBlocks", ctypes.POINTER(ctypes.c_uint32)),
        ("mRefFactors", ctypes.POINTER(ctypes.c_int)),
        ("mSpacing", ctypes.POINTER(ctypes.c_int))]

    fields = bPartition_fields_ + mPartition_fields_

    # Create the new class dynamically
    suffix = type_mapping['suffix']
    new_class = type(
        f'mPartitionGeneric_{suffix}',  # Class name with mem_type name appended
        (ctypes.Structure,),  # Base classes
        {
            '_fields_': fields,
            '__init__': mPartitionGeneric.__init__,
            '_help_load_api': mPartitionGeneric._help_load_api,
            '__str__': mPartitionGeneric.__str__,
        }
    )

    return new_class


mPartition_int8 = factory_mPartition(wp.int8)
mPartition_uint8 = factory_mPartition(wp.uint8)
mPartition_bool = factory_mPartition(wp.bool)

mPartition_int32 = factory_mPartition(wp.int32)
mPartition_uint32 = factory_mPartition(wp.uint32)

mPartition_int64 = factory_mPartition(wp.int64)
mPartition_uint64 = factory_mPartition(wp.uint64)

mPartition_float32 = factory_mPartition(wp.float32)
mPartition_float64 = factory_mPartition(wp.float64)


def register_builtins():
    supported_types = [(mPartition_int8, 'int8', wp.int8),
                       (mPartition_uint8, 'uint8', wp.uint8),

                       (mPartition_int32, 'int32', wp.int32),
                       (mPartition_uint32, 'uint32', wp.uint32),

                       (mPartition_int64, 'int64', wp.int64),
                       (mPartition_uint64, 'uint64', wp.uint64),

                       (mPartition_float32, 'float32', wp.float32),
                       (mPartition_float64, 'float64', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # register type
        wp.types.add_type(Partition, native_name=f"NeonMultiresPartition_{suffix}", has_binary_ctor=True)

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
