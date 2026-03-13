"""
Ngh_idx Module for Neon Computing Framework

This module provides the Ngh_idx class for representing stencil neighbor
offsets in 3D grids. Neighbor indices use 8-bit integers for compact storage.
"""

import ctypes
import typing


class Ngh_idx(ctypes.Structure):
    """
    A 3D neighbor index for stencil operations.
    
    Ngh_idx represents an offset to a neighboring cell in a stencil pattern.
    It uses 8-bit integers for compact storage, suitable for typical stencil
    offsets in the range [-127, 127].
    
    This is commonly used to define stencil patterns for operations that
    read from neighboring cells (e.g., finite differences, convolutions).
    
    This is a ctypes Structure that can be passed directly to C++ functions.
    
    Attributes:
        x (int8): X offset to neighbor.
        y (int8): Y offset to neighbor.
        z (int8): Z offset to neighbor.
    
    Example:
        >>> import neon
        >>> 
        >>> # Define a 7-point stencil (center + 6 face neighbors)
        >>> stencil = [
        ...     neon.Ngh_idx(0, 0, 0),   # center
        ...     neon.Ngh_idx(-1, 0, 0),  # -x neighbor
        ...     neon.Ngh_idx(1, 0, 0),   # +x neighbor
        ...     neon.Ngh_idx(0, -1, 0),  # -y neighbor
        ...     neon.Ngh_idx(0, 1, 0),   # +y neighbor
        ...     neon.Ngh_idx(0, 0, -1),  # -z neighbor
        ...     neon.Ngh_idx(0, 0, 1),   # +z neighbor
        ... ]
    """
    
    _fields_ = [("x", ctypes.c_int8),
                ("y", ctypes.c_int8),
                ("z", ctypes.c_int8)]

    def __init__(self, x: int, y: int, z: int):
        """
        Create a 3D neighbor offset.
        
        Args:
            x (int): X offset (-127 to 127).
            y (int): Y offset (-127 to 127).
            z (int): Z offset (-127 to 127).
        
        Example:
            >>> ngh = neon.Ngh_idx(1, 0, 0)  # +x neighbor
        """
        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        """Return the number of dimensions (always 3)."""
        return 3

    def __getitem__(self, index: int) -> int:
        """
        Get component by index.
        
        Args:
            index (int): 0 for x, 1 for y, 2 for z.
        
        Returns:
            int: The offset value.
        
        Raises:
            IndexError: If index is not 0, 1, or 2.
        """
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        raise IndexError("Index out of range")

    def to_wp_kernel_dim(self) -> typing.Tuple[int, int, int]:
        """
        Convert to tuple.
        
        Returns:
            Tuple[int, int, int]: (x, y, z) tuple.
        """
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        """Return string representation."""
        s = "<Ngh_idx: addr=%ld>" % (ctypes.addressof(self))
        s += f"\n\tx: {self.x}"
        s += f"\n\ty: {self.y}"
        s += f"\n\tz: {self.z}"
        return s

    def __eq__(self, other) -> bool:
        """Check equality with another Ngh_idx."""
        if not isinstance(other, Ngh_idx):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)


    @staticmethod
    def warp_register_builtins():
        import warp as wp
        # register type
        wp.types.add_type(Ngh_idx, native_name="NeonNghIdx")

        # create dense index
        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"x": wp.int8, "y": wp.int8, "z": wp.int8},
            value_type=Ngh_idx,
            missing_grad=True,
        )

        # create dense index
        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"idx": Ngh_idx, "x": wp.int8, "y": wp.int8, "z": wp.int8},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_ngh_idx",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_y",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_z",
            input_types={"idx": Ngh_idx},
            value_type=wp.int8,
            missing_grad=True,
        )

        # print dense index
        wp.context.add_builtin(
            "neon_print",
            input_types={"a": Ngh_idx},
            value_type=None,
            missing_grad=True,
        )
