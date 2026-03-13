"""
Backend Module for Neon Computing Framework

This module provides the Backend class, which manages computational resources
and device configurations for executing Neon operations across CPUs and GPUs.
"""

import ctypes
from enum import Enum
from typing import List
import warp as wp


import neon


class Backend(object):
    """
    Computational backend manager for Neon operations.
    
    The Backend class configures and manages the execution environment for Neon
    computations, including device selection (CPU/GPU), runtime configuration,
    and synchronization primitives.
    
    Supported runtimes:
        - ``Runtime.openmp``: OpenMP-based parallel execution on CPU (default)
        - ``Runtime.stream``: CUDA stream-based execution on GPU
        - ``Runtime.system``: System default (same as none)
    
    Attributes:
        backend_handle (ctypes.c_void_p): Handle to the C++ backend object.
        cuda_driver_handle (ctypes.c_void_p): Handle to the CUDA driver context.
        n_dev (int): Number of devices being used.
        dev_idx_list (List[int]): List of device indices.
        runtime (Runtime): The runtime configuration.
        neon_gate (neon.Gate): Interface to the C++ Neon library.
    
    Example:
        >>> import neon
        >>> 
        >>> # Create a single-GPU backend
        >>> backend = neon.Backend(runtime=neon.Backend.Runtime.stream, n_dev=1)
        >>> 
        >>> # Create a multi-GPU backend
        >>> backend = neon.Backend(
        ...     runtime=neon.Backend.Runtime.stream,
        ...     n_dev=2,
        ...     dev_idx_list=[0, 1]
        ... )
        >>> 
        >>> # Create a CPU backend with OpenMP
        >>> backend = neon.Backend(runtime=neon.Backend.Runtime.openmp)
        >>> 
        >>> # Synchronize all devices
        >>> backend.sync()
    
    Note:
        The backend must be created before any grids or fields can be allocated.
        Device indices must correspond to valid CUDA devices on the system.
    """
    
    class Runtime(Enum):
        """
        Enumeration of supported runtime configurations.
        
        Attributes:
            none: No specific runtime (system default).
            system: System default runtime (same as none).
            stream: CUDA stream-based GPU execution.
            openmp: OpenMP-based CPU parallel execution.
        """
        none = 0
        system = 0
        stream = 1
        openmp = 2

    def __init__(self,
                 runtime: Runtime = Runtime.openmp,
                 n_dev: int = 1,
                 dev_idx_list: List[int] = [0]):
        """
        Initialize a computational backend.
        
        Args:
            runtime (Runtime, optional): The runtime configuration to use.
                Defaults to Runtime.openmp for CPU execution.
            n_dev (int, optional): Number of devices to use. Defaults to 1.
            dev_idx_list (List[int], optional): List of device indices to use.
                Defaults to [0]. If n_dev > len(dev_idx_list), the list is
                automatically extended to [0, 1, ..., n_dev-1].
        
        Raises:
            Exception: If backend initialization fails (e.g., CUDA not available
                when using Runtime.stream, or invalid device indices).
        
        Example:
            >>> backend = neon.Backend(
            ...     runtime=neon.Backend.Runtime.stream,
            ...     n_dev=2,
            ...     dev_idx_list=[0, 1]
            ... )
        """

        self.backend_handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.cuda_driver_handle: ctypes.c_void_p = ctypes.c_void_p(0)

        self.n_dev = n_dev
        self.dev_idx_list = dev_idx_list
        self.runtime = runtime

        devices = {}
        devices['cuda:0'] = wp.get_device("cuda:0")
        try:
            self.neon_gate: neon = neon.Gate()
        except Exception as e:
            self.backend_handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))
        self.help_load_api()
        self.help_backend_new()

    def __del__(self):
        if self.backend_handle == 0:
            return
        self.help_backend_delete()
        pass

    def help_load_api(self):
        # ------------------------------------------------------------------
        # backend_new
        lib_obj = self.neon_gate.lib
        self.api_new = lib_obj.backend_new
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int)]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_delete
        self.api_delete = lib_obj.backend_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # backend_get_string
        # self.api_get_string = lib_obj.backend_get_string
        # self.api_get_string.argtypes = [self.neon_gate.handle_type]
        # self.api_get_string.restype = ctypes.c_char_p
        # ------------------------------------------------------------------
        # cuda_driver_new
        self.api_cuda_driver_new = lib_obj.cuda_driver_new
        self.api_cuda_driver_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                                     self.neon_gate.handle_type]
        self.api_cuda_driver_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # cuda_driver_delete
        self.api_cuda_driver_delete = lib_obj.cuda_driver_delete
        self.api_cuda_driver_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_cuda_driver_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # backend_sync
        self.api_sync = lib_obj.backend_sync
        self.api_sync.argtypes = [self.neon_gate.handle_type]
        self.api_sync.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # cuda_driver_delete
        self.api_info_print = lib_obj.backend_info_print
        self.api_info_print.argtypes = [self.neon_gate.handle_type]
        self.api_info_print.restype = ctypes.c_int
        # TODOMATT get num devices
        # TODOMATT get device type

    # def info(self):
    #     return  print(f"INFO Backend handle {hex(self.backend_handle.value)}")


    def help_backend_new(self):
        if self.backend_handle.value != ctypes.c_void_p(0).value:
            raise Exception(f'DBackend: Invalid handle {self.backend_handle}')

        if self.n_dev > len(self.dev_idx_list):
            self.dev_idx_list = list(range(self.n_dev))
        else:
            self.n_dev = len(self.dev_idx_list)

        # Loading the device list into a contiguous array
        dev_array = (ctypes.c_int * self.n_dev)(*self.dev_idx_list)

        res = self.api_new(ctypes.pointer(self.backend_handle),
                                            self.runtime.value,
                                            self.n_dev,
                                            dev_array)

        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

        res = self.api_cuda_driver_new(ctypes.pointer(self.cuda_driver_handle),
                                         self.backend_handle)

        if res != 0:
            raise Exception('DBackend: Failed to initialize backend')

        pass


    def help_backend_delete(self):
        if self.backend_handle == 0:
            return
        self.api_cuda_driver_delete(ctypes.pointer(self.cuda_driver_handle))
        res = self.api_delete(ctypes.pointer(self.backend_handle))
        if res != 0:
            raise Exception('Failed to delete backend')


    def get_num_devices(self) -> int:
        """
        Get the number of devices configured for this backend.
        
        Returns:
            int: The number of devices.
        
        Example:
            >>> backend = neon.Backend(n_dev=2)
            >>> backend.get_num_devices()
            2
        """
        return self.n_dev

    def get_warp_device_name(self) -> str:
        """
        Get the Warp device type string for this backend.
        
        Returns:
            str: 'cuda' for GPU backends, 'cpu' for CPU backends.
        
        Example:
            >>> gpu_backend = neon.Backend(runtime=neon.Backend.Runtime.stream)
            >>> gpu_backend.get_warp_device_name()
            'cuda'
        """
        if self.runtime == Backend.Runtime.stream:
            return 'cuda'
        else:
            return 'cpu'

    def sync(self) -> int:
        """
        Synchronize all devices managed by this backend.
        
        Blocks until all pending operations on all devices have completed.
        This is useful for ensuring data consistency before reading results
        or timing operations.
        
        Returns:
            int: 0 on success, non-zero on failure.
        
        Example:
            >>> backend.sync()  # Wait for all GPU operations to complete
        """
        return self.neon_gate.lib.backend_sync(self.backend_handle)

    def get_device_name(self, dev_idx: int) -> str:
        """
        Get the device name string for a specific device index.
        
        Args:
            dev_idx (int): The device index (0-based within this backend).
        
        Returns:
            str: Device name in the format 'cuda:N' or 'cpu:N' where N is
                the actual device ID from dev_idx_list.
        
        Example:
            >>> backend = neon.Backend(
            ...     runtime=neon.Backend.Runtime.stream,
            ...     dev_idx_list=[2, 3]
            ... )
            >>> backend.get_device_name(0)
            'cuda:2'
            >>> backend.get_device_name(1)
            'cuda:3'
        """
        if self.runtime == Backend.Runtime.stream:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cuda:{dev_id}"
        else:
            dev_id = self.dev_idx_list[dev_idx]
            return f"cpu:{dev_id}"

    def info_print(self) -> None:
        """
        Print backend information to stdout.
        
        Outputs details about the backend configuration including runtime type,
        number of devices, and device properties.
        """
        self.api_info_print(self.backend_handle)
