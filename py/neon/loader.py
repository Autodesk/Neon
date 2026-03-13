"""
Loader Module for Neon Computing Framework

This module provides the Loader class, which manages field access patterns
and kernel declarations within Neon containers. The Loader tracks read/write
dependencies for automatic optimization by the Skeleton.
"""

from enum import Enum
import neon


class Loader:
    """
    Manages field handles and kernel declarations for containers.
    
    The Loader is passed to container loading functions and provides methods
    to obtain read/write handles to fields. It tracks access patterns to enable
    automatic dependency analysis by the Skeleton.
    
    Attributes:
        execution (neon.Execution): Execution context (device/host).
        gpu_id (int): GPU device index for this loader.
        data_view (neon.DataView): Data view for field partitions.
        parsing (bool): Whether in parsing mode (collecting dependencies).
        tokens (List[Token]): Collected field access tokens.
        mres_level (int): Multi-resolution level (for mGrid).
        neon_grid: The grid associated with this loader.
        kernel: The declared kernel function.
    
    Example:
        >>> def my_loader(l: neon.Loader):
        ...     # Get read handle - marks field_a as input
        ...     a = l.get_read_handle(field_a)
        ...     
        ...     # Get write handle - marks field_b as output
        ...     b = l.get_write_handle(field_b)
        ...     
        ...     @neon.kernel(l)
        ...     def compute(idx):
        ...         b[idx, 0] = a[idx, 0] * 2.0
    
    Note:
        The Loader is typically not instantiated directly by users. It is
        created and passed by the Container during kernel compilation.
    """
    
    class Access(Enum):
        """
        Field access mode enumeration.
        
        Attributes:
            read: Field is read but not modified.
            write: Field is modified.
        """
        read = 1
        write = 2

    class Operation(Enum):
        """
        Computational operation pattern enumeration.
        
        Attributes:
            map: Point-wise operation (each cell independent).
            stencil: Stencil operation (reads from neighboring cells).
            stencil_up: Multi-resolution stencil reading finer level.
            stencil_down: Multi-resolution stencil reading coarser level.
        """
        map = 0
        stencil = 1
        stencil_up = 2
        stencil_down = 3

    class Discretization(Enum):
        """
        Grid discretization type enumeration.
        
        Attributes:
            cartesian: Standard Cartesian grid indexing.
            lattice: Lattice-based indexing (e.g., for LBM).
        """
        cartesian = 0
        lattice = 1

    class Token:
        """
        Records a field access pattern for dependency analysis.
        
        Attributes:
            field: The field being accessed.
            access (Access): Read or write access.
            operation (Operation): Type of operation.
            discretization (Discretization): Grid discretization type.
        """
        def __init__(self, field, access, operation, discretization):
            self.field = field
            self.access = access
            self.operation = operation
            self.discretization = discretization
            
        def get_field(self):
            """Return the accessed field."""
            return self.field
            
        def get_access(self):
            """Return the access mode (read/write)."""
            return self.access
            
        def get_operation(self):
            """Return the operation type."""
            return self.operation
            
        def get_discretization(self):
            """Return the discretization type."""
            return self.discretization

    def __init__(self,
                 execution: neon.Execution,
                 gpu_id: int,
                 data_view: neon.DataView,
                 parsing: bool = False):
        """
        Initialize a Loader.
        
        Args:
            execution (neon.Execution): Execution context (device/host).
            gpu_id (int): GPU device index.
            data_view (neon.DataView): Data view for field partitions.
            parsing (bool, optional): If True, collect dependency tokens.
                Defaults to False.
        """

        self.mres_level = None
        self.execution = execution
        self.gpu_id = gpu_id
        self.data_view = data_view
        self.parsing = parsing

        self.tokens = []

        self.kernel = None
        self.neon_grid = None


    def get_read_handle(
        self,
        neon_field,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian
    ):
        """
        Get a read handle (partition) for a field.
        
        Marks the field as a read dependency and returns a partition object
        that can be used to read field values in the kernel.
        
        Args:
            neon_field: The Neon field to read from.
            operation (Operation, optional): The operation pattern.
                Defaults to Operation.map.
            discretization (Discretization, optional): Grid discretization.
                Defaults to Discretization.cartesian.
        
        Returns:
            Partition object for reading field data.
        
        Example:
            >>> def loader(l: neon.Loader):
            ...     a = l.get_read_handle(field_a)
            ...     # Use a[idx, component] in kernel to read values
        """
        if self.parsing:
            access = Loader.Access.read
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)
        return partition

    def get_write_handle(
        self,
        neon_field,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian
    ):
        """
        Get a write handle (partition) for a field.
        
        Marks the field as a write dependency and returns a partition object
        that can be used to write field values in the kernel.
        
        Args:
            neon_field: The Neon field to write to.
            operation (Operation, optional): The operation pattern.
                Defaults to Operation.map.
            discretization (Discretization, optional): Grid discretization.
                Defaults to Discretization.cartesian.
        
        Returns:
            Partition object for writing field data.
        
        Example:
            >>> def loader(l: neon.Loader):
            ...     b = l.get_write_handle(field_b)
            ...     # Use b[idx, component] = value in kernel to write values
        """
        if self.parsing:
            access = Loader.Access.write
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.execution,
            self.gpu_id,
            self.data_view)

        return partition

    def get_mres_write_handle(
        self,
        neon_field,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian
    ):
        """
        Get a write handle for a multi-resolution field at the current level.
        
        Similar to ``get_write_handle()`` but for multi-resolution grids (mGrid),
        accessing the field at the level set by ``set_mres_grid()``.
        
        Args:
            neon_field: The mField to write to.
            operation (Operation, optional): The operation pattern.
            discretization (Discretization, optional): Grid discretization.
        
        Returns:
            Partition object for writing at the specified resolution level.
        """
        if self.parsing:
            access = Loader.Access.write
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.mres_level,
            self.execution,
            self.gpu_id,
            self.data_view)

        return partition

    def get_mres_read_handle(
        self,
        neon_field,
        operation: Operation = Operation.map,
        discretization: Discretization = Discretization.cartesian
    ):
        """
        Get a read handle for a multi-resolution field at the current level.
        
        Similar to ``get_read_handle()`` but for multi-resolution grids (mGrid),
        accessing the field at the level set by ``set_mres_grid()``.
        
        Args:
            neon_field: The mField to read from.
            operation (Operation, optional): The operation pattern.
            discretization (Discretization, optional): Grid discretization.
        
        Returns:
            Partition object for reading at the specified resolution level.
        """
        if self.parsing:
            access = Loader.Access.read
            token = Loader.Token(neon_field, access, operation, discretization)
            self.tokens.append(token)

        partition = neon_field.get_partition(
            self.mres_level,
            self.execution,
            self.gpu_id,
            self.data_view
        )

        return partition

    def set_grid(self, grid) -> None:
        """
        Associate a grid with this loader.
        
        Args:
            grid: The Neon grid (dGrid, bGrid, etc.).
        """
        self.neon_grid = grid

    def set_mres_grid(self, grid, level: int) -> None:
        """
        Associate a multi-resolution grid and level with this loader.
        
        Args:
            grid: The mGrid instance.
            level (int): The resolution level to operate on.
        """
        self.neon_grid = grid
        self.mres_level = level

    def get_mres_level(self) -> int:
        """
        Get the current multi-resolution level.
        
        Returns:
            int: The resolution level, or None if not set.
        """
        return self.mres_level

    def _retrieve_grid(self):
        """Internal: Get the associated grid."""
        return self.neon_grid

    def declare_kernel(self, kernel) -> None:
        """
        Register a kernel function with this loader.
        
        This is called by the ``@neon.kernel`` decorator to associate
        the kernel function with this loader.
        
        Args:
            kernel: The Warp kernel function.
        """
        self.kernel = kernel

    def _retrieve_compute_lambda(self):
        """Internal: Get the registered kernel function."""
        return self.kernel

    def _get_tokens(self):
        """Internal: Get collected access tokens for dependency analysis."""
        return self.tokens

    def get_device_id(self) -> int:
        """
        Get the GPU device ID for this loader.
        
        Returns:
            int: The GPU device index.
        """
        return self.gpu_id
