"""
Skeleton Module for Neon Computing Framework

This module provides the Skeleton class, which manages the orchestration and
execution of sequences of computational containers (kernels) in the Neon framework.
"""

import ctypes
from typing import List
from neon import SkeletonConfig
import neon


class Skeleton(object):
    """
    Orchestrates execution of computational container sequences.
    
    The Skeleton class manages the scheduling, dependency analysis, and execution
    of sequences of Neon containers (computational kernels). It provides automatic
    optimization of execution order based on data dependencies and supports
    visualization of the execution graph.
    
    Key features:
        - Automatic dependency analysis between containers
        - Optimized execution scheduling
        - Support for overlap computation and communication (OCC)
        - DOT graph export for visualization
    
    Attributes:
        skeleton_handle (ctypes.c_void_p): Handle to the C++ skeleton object.
        backend (neon.Backend): The computational backend for execution.
        neon_gate (neon.Gate): Interface to the C++ Neon library.
        containers (List[neon.Container]): List of containers in the current sequence.
    
    Example:
        >>> import neon
        >>> 
        >>> # Create backend and grid
        >>> backend = neon.Backend(runtime=neon.Backend.Runtime.stream)
        >>> grid = neon.dGrid(backend=backend, dim=neon.Index_3d(64, 64, 64))
        >>> 
        >>> # Create some containers (kernels)
        >>> container1 = create_my_kernel(grid)
        >>> container2 = create_another_kernel(grid)
        >>> 
        >>> # Create skeleton and define sequence
        >>> skeleton = neon.Skeleton(backend=backend)
        >>> skeleton.sequence("my_computation", [container1, container2])
        >>> 
        >>> # Execute the sequence
        >>> skeleton.run()
        >>> 
        >>> # Export execution graph for visualization
        >>> skeleton.ioToDot("graph.dot", "MyComputation")
    """
    
    def __init__(self, backend: neon.Backend):
        """
        Initialize a Skeleton for orchestrating container execution.
        
        Args:
            backend (neon.Backend): The computational backend to use for execution.
        
        Raises:
            Exception: If skeleton initialization fails.
        
        Example:
            >>> backend = neon.Backend()
            >>> skeleton = neon.Skeleton(backend=backend)
        """

        self.skeleton_handle: ctypes.c_void_p = ctypes.c_void_p(0)
        self.backend = backend

        try:
            self.neon_gate: neon.Gate = neon.Gate()
        except Exception as e:
            self.skeleton_handle = ctypes.c_void_p(0)
            raise Exception('Failed to initialize PyNeon: ' + str(e))

        self.help_load_api()
        self.help_skeleton_new()

    def __del__(self):
        if self.skeleton_handle == 0:
            return
        self.help_skeleton_delete()
        pass

    def help_load_api(self):
        lib_obj = self.neon_gate.lib
        # ------------------------------------------------------------------
        # neon_skeleton_new
        self.api_new = lib_obj.neon_skeleton_new
        self.api_new.argtypes = [ctypes.POINTER(self.neon_gate.handle_type),
                                 self.neon_gate.handle_type]
        self.api_new.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_delete
        self.api_delete = lib_obj.neon_skeleton_delete
        self.api_delete.argtypes = [ctypes.POINTER(self.neon_gate.handle_type)]
        self.api_delete.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_sequence
        self.api_sequence = lib_obj.neon_skeleton_sequence
        self.api_sequence.argtypes = [self.neon_gate.handle_type,
                                      ctypes.c_char_p,
                                      ctypes.c_int,
                                      ctypes.POINTER(self.neon_gate.handle_type),
                                      ctypes.c_int]
        self.api_sequence.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_run
        self.api_run = lib_obj.neon_skeleton_run
        self.api_run.argtypes = [self.neon_gate.handle_type]
        self.api_run.restype = ctypes.c_int
        # ------------------------------------------------------------------
        # neon_skeleton_ioToDot
        self.api_ioToDot = lib_obj.neon_skeleton_ioToDot
        self.api_ioToDot.argtypes = [self.neon_gate.handle_type,
                                 ctypes.c_char_p,
                                 ctypes.c_char_p,
                                 ctypes.c_int]
        self.api_ioToDot.restype = ctypes.c_int

    def help_skeleton_new(self):
        if self.skeleton_handle.value != ctypes.c_void_p(0).value:
            raise Exception(f'Skeleton: Invalid handle {self.skeleton_handle}')

        res = self.api_new(ctypes.pointer(self.skeleton_handle),
                                                 self.backend.backend_handle)

        if res != 0:
            raise Exception('Backend: Failed to initialize backend')

    def help_skeleton_delete(self):
        if self.skeleton_handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.skeleton_handle))
        if res != 0:
            raise Exception('Failed to delete backend')

    def sequence(
        self,
        name: str,
        containers: List[neon.Container],
        occ: neon.SkeletonConfig.OCC = neon.SkeletonConfig.OCC.none()
    ) -> None:
        """
        Define a named sequence of containers to execute.
        
        Analyzes the data dependencies between containers and creates an optimized
        execution plan. Multiple sequences can be defined on the same skeleton.
        
        Args:
            name (str): A unique name for this sequence.
            containers (List[neon.Container]): Ordered list of containers (kernels)
                to execute. The skeleton will analyze dependencies and may reorder
                or parallelize execution where safe.
            occ (SkeletonConfig.OCC, optional): Overlap computation and communication
                strategy. Defaults to OCC.none() (no overlap).
        
        Example:
            >>> skeleton.sequence("timestep", [
            ...     compute_forces,
            ...     update_velocities,
            ...     update_positions
            ... ])
        
        Note:
            Container dependencies are automatically inferred from field read/write
            patterns specified during container creation.
        """
        self.containers = containers
        self.handle_list = (ctypes.c_void_p * len(containers))()
        for i in range(len(self.handle_list)):
            self.handle_list[i] = containers[i].container_handle
        from .logging import logger
        logger.debug(f"handle_list {self.handle_list}")
        logger.debug(f"handle_list[0] {hex(self.handle_list[0])}")
        self.api_sequence(self.skeleton_handle,
                          name.encode('utf-8'),
                          len(self.handle_list),
                          self.handle_list,
                          occ.value)

    def run(self) -> None:
        """
        Execute the defined container sequence.
        
        Runs all containers in the sequence according to the optimized execution
        plan. This method blocks until all operations complete.
        
        Example:
            >>> skeleton.run()  # Execute the computation
        
        Note:
            A sequence must be defined via ``sequence()`` before calling ``run()``.
        """
        self.api_run(self.skeleton_handle)

    def ioToDot(self, filename: str, graph_name: str) -> None:
        """
        Export the execution graph to a DOT file for visualization.
        
        Creates a DOT format file representing the container dependency graph,
        which can be visualized using Graphviz or similar tools.
        
        Args:
            filename (str): Path to the output DOT file.
            graph_name (str): Name to use for the graph in the DOT file.
        
        Example:
            >>> skeleton.ioToDot("execution_graph.dot", "MySimulation")
            >>> # Then visualize with: dot -Tpng execution_graph.dot -o graph.png
        """
        self.api_ioToDot(self.skeleton_handle,
                         filename.encode('utf-8'),
                         graph_name.encode('utf-8'),
                         0)
