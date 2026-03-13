"""
Index_3d Module for Neon Computing Framework

This module provides the Index_3d class for representing 3D indices
and coordinates in Neon grids.
"""

import ctypes
import typing


class Index_3d(ctypes.Structure):
    """
    A 3D integer index for grid coordinates.
    
    Index_3d represents a point or dimension in 3D space using integer
    coordinates. It is used throughout Neon for specifying grid dimensions,
    cell indices, and stencil offsets.
    
    This is a ctypes Structure that can be passed directly to C++ functions.
    
    Attributes:
        x (int): X coordinate.
        y (int): Y coordinate.
        z (int): Z coordinate.
    
    Example:
        >>> import neon
        >>> 
        >>> # Create grid dimensions
        >>> dim = neon.Index_3d(64, 64, 64)
        >>> print(dim.x, dim.y, dim.z)
        64 64 64
        >>> 
        >>> # Use as a sequence
        >>> print(list(dim))
        [64, 64, 64]
        >>> print(dim[0])  # x component
        64
        >>> 
        >>> # Stencil offset
        >>> neighbor = neon.Index_3d(1, 0, 0)  # +x neighbor
    """
    
    _fields_ = [("x", ctypes.c_int32),
                ("y", ctypes.c_int32),
                ("z", ctypes.c_int32)]

    def __init__(self, x: int, y: int, z: int):
        """
        Create a 3D index.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            z (int): Z coordinate.
        
        Example:
            >>> idx = neon.Index_3d(10, 20, 30)
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
            int: The coordinate value.
        
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
        Convert to Warp kernel dimension tuple.
        
        Returns:
            Tuple[int, int, int]: (x, y, z) tuple for Warp kernel launch.
        """
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        """Return string representation."""
        s = f"({self.x}, "
        s += f"{self.y}, "
        s += f"{self.z})"
        s += "<Index_3d: addr=%ld>" % (ctypes.addressof(self))
        return s

    def __eq__(self, other) -> bool:
        """Check equality with another Index_3d."""
        if not isinstance(other, Index_3d):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    @staticmethod
    def warp_register_builtins():
        import warp as wp

        # register type
        wp.types.add_type(Index_3d, native_name="NeonIndex3d")

        # create dense index
        wp.context.add_builtin(
            "neon_idx_3d",
            input_types={"x": int, "y": int, "z": int},
            value_type=Index_3d,
            missing_grad=True,
        )

        # create dense index
        wp.context.add_builtin(
            "neon_init",
            input_types={"idx": Index_3d, "x": int, "y": int, "z": int},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_x",
            input_types={"idx": Index_3d},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_y",
            input_types={"idx": Index_3d},
            value_type=int,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_z",
            input_types={"idx": Index_3d},
            value_type=int,
            missing_grad=True,
        )

        # print dense index
        wp.context.add_builtin(
            "neon_print",
            input_types={"a": Index_3d},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_is_equal",
            input_types={"idx": Index_3d, "x": int, "y": int, "z": int},
            value_type=bool,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_cuda_info",
            input_types={},
            value_type=None,
            missing_grad=True,
        )

        wp.context.add_builtin(
            "neon_get_component",
            input_types={"idx": Index_3d, "component": int},
            value_type=int,
            missing_grad=True,
        )
