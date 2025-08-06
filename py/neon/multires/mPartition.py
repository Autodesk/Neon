"""
Multi-resolution Partition Implementation for Neon Computing Framework

This module provides the mPartition classes, which represent partitioned data structures
for multi-resolution parallel computing with Warp integration. The partitions enable
efficient computation across multiple resolution levels with hierarchical relationships
between parent and child levels.

The module uses a factory pattern to create type-specific partition classes that are
compatible with Warp's kernel system and provide GPU-accelerated multi-resolution
computation capabilities.

Key Features:
- Dynamic type-specific partition class generation
- Multi-resolution hierarchy support (parent/child relationships)
- Warp kernel integration with built-in functions
- Memory-efficient sparse data structures
- Cross-level data access and interpolation
- GPU-accelerated parallel processing

"""

import ctypes
import typing
from typing import Type, Any, Dict, List, Tuple

import neon
import warp as wp


class mPartitionGeneric(ctypes.Structure):
    """
    Generic base class for multi-resolution partitions.
    
    This class serves as the foundation for all type-specific multi-resolution
    partitions. It defines the common structure and interface used across all
    data types in the multi-resolution hierarchy.
    
    The partition represents a portion of the computational domain at a specific
    resolution level, with connections to parent (coarser) and child (finer)
    levels. This enables adaptive mesh refinement and multi-scale computations.
    
    Attributes:
        mCardinality (int): Number of components per data element
        mMem (POINTER): Main data memory for this partition
        mStencilNghIndex (POINTER): Stencil neighborhood indices
        mBlockConnectivity (POINTER): Inter-block connectivity information
        mMask (POINTER): Active element mask for sparse computation
        mOrigin (POINTER): Origin coordinates for this partition
        mSetIdx (int): Set index for parallel execution
        mMultiResDiscreteIdxSpacing (int): Spacing between discrete indices
        mDomainSize (Index_3d): Size of the computational domain
        mLevel (int): Resolution level (0 = finest, higher = coarser)
        mMemParent (POINTER): Parent level data memory
        mMemChild (POINTER): Child level data memory
        mParentBlockID (POINTER): Parent block identifiers
        mMaskLowerLevel (POINTER): Mask for lower (finer) level
        mMaskUpperLevel (POINTER): Mask for upper (coarser) level
        mChildBlockID (POINTER): Child block identifiers
        mParentNeighbourBlocks (POINTER): Parent neighbor block information
        mRefFactors (POINTER): Refinement factors between levels
        mSpacing (POINTER): Spatial spacing information
    """

    def __init__(self):
        """
        Initialize the multi-resolution partition.
        
        Sets up the Neon API gateway for accessing C++ backend functions.
        This is called automatically when creating partition instances.
        """
        self._help_load_api()

    def __str__(self) -> str:
        """
        Generate detailed string representation of the partition structure.
        
        Returns:
            str: Comprehensive representation showing all partition fields
                 with their memory addresses and offsets for debugging
        """
        str_repr = f"<mPartition: addr={ctypes.addressof(self):#x}>"
        str_repr += f"\n\tmCardinality: {self.mCardinality}"
        str_repr += f"\n\tmMem: {self.mMem}"
        str_repr += f"\n\tmStencilNghIndex: {self.mStencilNghIndex}"
        str_repr += f"\n\tmBlockConnectivity: {self.mBlockConnectivity}"
        str_repr += f"\n\tmMask: {self.mMask}"
        str_repr += f"\n\tmOrigin: {self.mOrigin}"
        str_repr += f"\n\tmSetIdx: {self.mSetIdx}"
        str_repr += f"\n\tmMultiResDiscreteIdxSpacing: {self.mMultiResDiscreteIdxSpacing}"
        str_repr += f"\n\tmDomainSize: {self.mDomainSize}"
        str_repr += f"\n\tmLevel: {self.mLevel}"
        str_repr += f"\n\tmMemParent: {self.mMemParent}"
        str_repr += f"\n\tmMemChild: {self.mMemChild}"
        str_repr += f"\n\tmParentBlockID: {self.mParentBlockID}"
        str_repr += f"\n\tmMaskLowerLevel: {self.mMaskLowerLevel}"
        str_repr += f"\n\tmMaskUpperLevel: {self.mMaskUpperLevel}"
        str_repr += f"\n\tmChildBlockID: {self.mChildBlockID}"
        str_repr += f"\n\tmParentNeighbourBlocks: {self.mParentNeighbourBlocks}"
        str_repr += f"\n\tmRefFactors: {self.mRefFactors}"
        str_repr += f"\n\tmSpacing: {self.mSpacing}"
        return str_repr

    def _help_load_api(self) -> None:
        """
        Load the Neon API gateway for accessing C++ backend functions.
        
        This method initializes the connection to the Neon C++ library,
        enabling access to native multi-resolution computation functions.
        """
        self.neon_gate: neon.Gate = neon.Gate()


def factory_mPartition(dtype) -> Type[ctypes.Structure]:
    """
    Factory function to create type-specific multi-resolution partition classes.
    
    This function dynamically generates a new partition class with memory fields
    typed according to the specified data type. This enables type-safe access
    to partition data while maintaining compatibility with Warp's kernel system.
    
    The factory pattern allows for efficient code generation and type specialization
    without manual duplication of partition structures for each data type.
    
    Args:
        dtype: Warp data type for the partition elements (e.g., wp.float32, wp.int32).
               This determines the type of data stored in the partition's memory fields.
    
    Returns:
        Type[ctypes.Structure]: A new dynamically created class that inherits from
                               ctypes.Structure with the appropriate type-specific
                               memory field definitions.
    
    Example:
        >>> FloatPartition = factory_mPartition(wp.float32)
        >>> partition = FloatPartition()
        >>> # partition.mMem now points to float32 data
    
    Note:
        The generated class includes all the multi-resolution hierarchy fields
        needed for parent-child relationships, connectivity, and spatial indexing.
    """
    # Get type mapping for the specified data type
    neon_gate: neon.Gate = neon.Gate()
    type_mapping = neon_gate.get_type_mapping(dtype)

    # Define base partition fields (inherited from block partition structure)
    bPartition_fields_ = [
        ("mCardinality", ctypes.c_int),           # Number of components per element
        ("mMem", ctypes.POINTER(ctypes.c_int)),   # Main data memory pointer
        ("mStencilNghIndex", ctypes.POINTER(ctypes.c_int)),     # Stencil neighbor indices
        ("mBlockConnectivity", ctypes.POINTER(ctypes.c_uint32)), # Block connectivity data
        ("mMask", ctypes.POINTER(ctypes.c_uint32)),             # Active element mask
        ("mOrigin", ctypes.POINTER(neon.Index_3d)),             # Partition origin coordinates
        ("mSetIdx", ctypes.c_int),                              # Set index for execution
        ("mMultiResDiscreteIdxSpacing", ctypes.c_int),          # Multi-res index spacing
        ("mDomainSize", neon.Index_3d)                          # Domain size
    ]

    # Define multi-resolution specific fields
    mPartition_fields_ = [
        ("mLevel", ctypes.c_int),                               # Resolution level
        ("mMemParent", ctypes.POINTER(ctypes.c_int)),           # Parent level memory
        ("mMemChild", ctypes.POINTER(ctypes.c_int)),            # Child level memory
        ("mParentBlockID", ctypes.POINTER(ctypes.c_uint32)),    # Parent block IDs
        ("mMaskLowerLevel", ctypes.POINTER(ctypes.c_uint32)),   # Lower level mask
        ("mMaskUpperLevel", ctypes.POINTER(ctypes.c_uint32)),   # Upper level mask
        ("mChildBlockID", ctypes.POINTER(ctypes.c_uint32)),     # Child block IDs
        ("mParentNeighbourBlocks", ctypes.POINTER(ctypes.c_uint32)), # Parent neighbors
        ("mRefFactors", ctypes.POINTER(ctypes.c_int)),          # Refinement factors
        ("mSpacing", ctypes.POINTER(ctypes.c_int))              # Spacing information
    ]

    # Combine all fields into complete structure definition
    fields = bPartition_fields_ + mPartition_fields_

    # Dynamically create the type-specific partition class
    suffix = type_mapping['suffix']
    new_class = type(
        f'mPartitionGeneric_{suffix}',  # Class name with type suffix
        (ctypes.Structure,),            # Inherit from ctypes.Structure
        {
            '_fields_': fields,                                 # Field definitions
            '__init__': mPartitionGeneric.__init__,             # Constructor
            '_help_load_api': mPartitionGeneric._help_load_api, # API loader
            '__str__': mPartitionGeneric.__str__,               # String representation
        }
    )

    return new_class

# Pre-defined partition types for common data types
# These are ready-to-use partition classes for standard numeric types

# Integer types
mPartition_int8 = factory_mPartition(wp.int8)      # 8-bit signed integer partitions
mPartition_uint8 = factory_mPartition(wp.uint8)    # 8-bit unsigned integer partitions
mPartition_bool = factory_mPartition(wp.bool)      # Boolean value partitions

mPartition_int32 = factory_mPartition(wp.int32)    # 32-bit signed integer partitions
mPartition_uint32 = factory_mPartition(wp.uint32)  # 32-bit unsigned integer partitions

mPartition_int64 = factory_mPartition(wp.int64)    # 64-bit signed integer partitions
mPartition_uint64 = factory_mPartition(wp.uint64)  # 64-bit unsigned integer partitions

# Floating-point types
mPartition_float32 = factory_mPartition(wp.float32) # 32-bit floating-point partitions
mPartition_float64 = factory_mPartition(wp.float64) # 64-bit floating-point partitions


def register_builtins() -> None:
    """
    Register all multi-resolution partition types and their built-in functions with Warp.
    
    This function registers the partition types with Warp's type system and adds
    all the built-in functions that can be used within Warp kernels for multi-resolution
    computations. The built-ins provide efficient GPU-accelerated operations for:
    
    - Basic data access (read/write)
    - Multi-resolution hierarchy navigation (parent/child access)
    - Neighborhood operations with stencils
    - Level-specific operations and queries
    - Inter-level data transfer and interpolation
    
    The registration process makes these functions available for use in @wp.kernel
    decorated functions, enabling high-performance multi-resolution computations
    on both CPU and GPU.
    
    Built-in Function Categories:
    - Data Access: neon_read, neon_write, neon_cardinality
    - Neighborhood: neon_ngh_idx, neon_read_ngh, neon_write_ngh
    - Multi-resolution: neon_read_parent, neon_read_child, neon_has_parent, neon_has_child
    - Spatial: neon_global_idx, neon_level, neon_refinement_factor, neon_spacing
    - Validation: neon_is_valid, neon_has_finer_ngh
    - Utilities: neon_partition_id, neon_device_id, neon_print_log
    
    Note:
        This function should be called once during module initialization to ensure
        all partition types and built-ins are available for Warp kernel usage.
    """
    supported_types = [(mPartition_int8, 'int8', wp.int8),
                       (mPartition_uint8, 'uint8', wp.uint8),

                       (mPartition_int32, 'int32', wp.int32),
                       (mPartition_uint32, 'uint32', wp.uint32),

                       (mPartition_int64, 'int64', wp.int64),
                       (mPartition_uint64, 'uint64', wp.uint64),

                       (mPartition_float32, 'float32', wp.float32),
                       (mPartition_float64, 'float64', wp.float64)]

    for Partition, suffix, Type in supported_types:
        # Register partition type with Warp's type system
        wp.types.add_type(Partition, native_name=f"NeonMultiresPartition_{suffix}", has_binary_ctor=True)

        # === Basic Data Access Built-ins ===
        
        wp.context.add_builtin(
            "neon_read",
            input_types={"partition": Partition,     # Multi-resolution partition
                         'idx': neon.block.bIndex,   # Block-local index
                         "card": int},               # Component index (for vector data)
            value_type=Type,                         # Returns data of partition's type
            missing_grad=True,                       # No gradient computation
            doc="""Read data from partition at specified index and component."""
        )

        wp.context.add_builtin(
            "neon_write",
            input_types={"partition": Partition,     # Multi-resolution partition
                         'idx': neon.block.bIndex,   # Block-local index
                         "card": int,                # Component index
                         "value": Type},             # Value to write
            value_type=None,                         # No return value
            missing_grad=True,
            doc="""Write data to partition at specified index and component."""
        )

        wp.context.add_builtin(
            "neon_cardinality",
            input_types={"partition": Partition},    # Multi-resolution partition
            value_type=int,                          # Returns number of components
            missing_grad=True,
            doc="""Get the number of components per element in the partition."""
        )

        wp.context.add_builtin(
            "neon_is_valid",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx},
            value_type=neon.block.bIndex,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_ngh",
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
            "neon_write_ngh",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx,
                         "card": wp.int32,
                         'value': Type},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_lbm_read_coarser_ngh",
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
        ###### Multi-resolution specific builtins
        wp.context.add_builtin(
            "neon_read_child",
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
            "neon_get_child",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx},
            value_type=neon.block.bIndex,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_child",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'card': wp.int32},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_has_child",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_has_finer_ngh",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_parent",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'card': wp.int32},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_write_parent",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'card': wp.int32,
                         'value': Type},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_has_parent",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex},
            value_type=wp.bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_getUncle",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx},
            value_type=neon.block.bIndex,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_mres_lbm_store_op",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'card': wp.int32,
                         'ngh_idx': neon.Ngh_idx,
                         'value': Type},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_coarser_ngh",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx,
                         'card': wp.int32,
                         'alternative': Type,
                         'is_valid': wp.bool},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_read_coarser_ngh",
            input_types={"partition": Partition,
                         'idx': neon.block.bIndex,
                         'ngh_idx': neon.Ngh_idx,
                         'card': wp.int32},
            value_type=Type,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_refinement_factor",
            input_types={"level": wp.int32},
            value_type=wp.int32,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_spacing",
            input_types={"level": wp.int32},
            value_type=wp.int32,
            missing_grad=True,
        )


        wp.context.add_builtin(
            "neon_level",
            input_types={"partition": Partition},
            value_type=wp.int32,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_print_log",
            input_types={"partition": Partition, "masterOnly": wp.bool},
            value_type=None,
            missing_grad=True,
        )

@wp.func
def neon_get_type(partition: typing.Any) -> typing.Any:
    """
    Get the Warp data type of a multi-resolution partition.
    
    This utility function determines the data type of a partition at compile time
    using Warp's static type introspection. It's useful for writing generic kernels
    that can work with partitions of different data types.
    
    Args:
        partition: A multi-resolution partition of any supported type
        
    Returns:
        The corresponding Warp data type (wp.int8, wp.float32, etc.)
        
    Note:
        This function uses wp.static() for compile-time type resolution,
        making it suitable for use in Warp kernels where runtime type
        checking is not available.
        
    Example:
        ```python
        @wp.kernel
        def process_partition(partition: mPartition_float32):
            data_type = neon_get_type(partition)  # Returns wp.float32
            # Use data_type for type-aware processing...
        ```
    
    TODO: Move this function to wp.neon namespace for consistency with other Neon functions.
    """
    if wp.static(isinstance(partition, mPartition_int8)):
        return wp.int8
    elif wp.static(isinstance(partition, mPartition_uint8)):
        return wp.uint8
    elif wp.static(isinstance(partition, mPartition_bool)):
        return wp.bool
    elif wp.static(isinstance(partition, mPartition_int32)):
        return wp.int32
    elif wp.static(isinstance(partition, mPartition_uint32)):
        return wp.uint32
    elif wp.static(isinstance(partition, mPartition_int64)):
        return wp.int64
    elif wp.static(isinstance(partition, mPartition_uint64)):
        return wp.uint64
    elif wp.static(isinstance(partition, mPartition_float32)):
        return wp.float32
    elif wp.static(isinstance(partition, mPartition_float64)):
        return wp.float64