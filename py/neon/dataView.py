"""
DataView Module for Neon Computing Framework

This module provides the DataView class, which specifies which subset of
grid cells to process during kernel execution.
"""

import ctypes
from enum import Enum


class DataView(ctypes.Structure):
    """
    Specifies which subset of grid data to process.
    
    DataView allows kernels to operate on different subsets of the computational
    domain. This is particularly useful for overlapping computation with
    communication in multi-GPU setups, where internal cells can be processed
    while boundary data is being exchanged.
    
    Data views:
        - ``standard``: All active cells (default)
        - ``internal``: Only cells that don't depend on halo data
        - ``boundary``: Only cells that require halo data
    
    This is a ctypes Structure that can be passed directly to C++ functions.
    
    Example:
        >>> import neon
        >>> 
        >>> # Process all cells (default)
        >>> container.run(stream_idx=0, data_view=neon.DataView.standard())
        >>> 
        >>> # Process only internal cells (for computation/communication overlap)
        >>> container.run(stream_idx=0, data_view=neon.DataView.internal())
        >>> 
        >>> # Process only boundary cells
        >>> container.run(stream_idx=0, data_view=neon.DataView.boundary())
    
    Note:
        Use the static factory methods to create instances.
    """
    
    _fields_ = [("data_view", ctypes.c_char)]

    class Values(Enum):
        """
        Enumeration of data view types.
        
        Attributes:
            standard: All active cells.
            internal: Cells not requiring halo data.
            boundary: Cells requiring halo data.
        """
        standard = 0
        internal = 1
        boundary = 2

    def __init__(self, data_view: 'DataView.Values'):
        """
        Initialize a DataView.
        
        Args:
            data_view (Values): The data view type.
        
        Note:
            Prefer using ``DataView.standard()``, ``DataView.internal()``,
            or ``DataView.boundary()`` factory methods.
        """
        if data_view == DataView.Values.standard:
            self.data_view = ctypes.c_char(b'\x00')
        elif data_view == DataView.Values.internal:
            self.data_view = ctypes.c_char(b'\x01')
        elif data_view == DataView.Values.boundary:
            self.data_view = ctypes.c_char(b'\x02')

    def __str__(self):
        str_repr = "<DDDData_view: addr=%ld, sizeof %ld>" % (ctypes.addressof(self), ctypes.sizeof(self))
        if self.data_view == ctypes.c_char(b'\x00'):
            str_repr += f"\n\tdataView: {'standard'}"
        elif self.data_view == ctypes.c_char(b'\x01'):
            str_repr += f"\n\tdataView: {'internal'}"
        elif self.data_view == ctypes.c_char(b'\x02'):
            str_repr += f"\n\tdataView: {'boundary'}"
        return str_repr

    @property
    def value(self):
        return self.data_view

    @value.setter
    def value(self, data_view: Values):
        self.data_view = data_view

    @staticmethod
    def standard() -> 'DataView':
        """
        Create a standard DataView (all active cells).
        
        Returns:
            DataView: A DataView for processing all cells.
        
        Example:
            >>> dv = neon.DataView.standard()
        """
        return DataView(DataView.Values.standard)

    @staticmethod
    def internal() -> 'DataView':
        """
        Create an internal DataView (cells not requiring halo data).
        
        Returns:
            DataView: A DataView for processing internal cells only.
        
        Example:
            >>> dv = neon.DataView.internal()
        """
        return DataView(DataView.Values.internal)

    @staticmethod
    def boundary() -> 'DataView':
        """
        Create a boundary DataView (cells requiring halo data).
        
        Returns:
            DataView: A DataView for processing boundary cells only.
        
        Example:
            >>> dv = neon.DataView.boundary()
        """
        return DataView(DataView.Values.boundary)

    @staticmethod
    def from_int(v: int) -> 'DataView':
        """
        Create a DataView from an integer value.
        
        Args:
            v (int): 0 for standard, 1 for internal, 2 for boundary.
        
        Returns:
            DataView: The corresponding DataView.
        
        Raises:
            Exception: If v is not 0, 1, or 2.
        """
        if v == 0:
            return DataView(DataView.Values.standard)
        if v == 1:
            return DataView(DataView.Values.internal)
        if v == 2:
            return DataView(DataView.Values.boundary)
        raise Exception('Invalid DataView value')


    def __eq__(self, other):
        if not isinstance(other, DataView):
            return NotImplemented
        return self.data_view == other.data_view


    @staticmethod
    def warp_register_builtins():
        import warp as wp

        # register type
        wp.types.add_type(DataView, native_name="NeonDataView")

        # print
        wp.context.add_builtin(
            "NeonDataView_print",
            input_types={"a": DataView},
            value_type=None,
            missing_grad=True,
        )
