import ctypes

import neon
import warp as wp


# from .bPartition import bPartitionInt as bPartitionInt


class bField(object):
    def __init__(self,
                 neon_gate: neon.Gate,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 dtype,
                 py_grid,
                 ):

        self.dtype = dtype
        if grid_handle == 0:
            raise Exception('bField: Invalid handle')

        self.neon_gate:neon.Gate =  neon_gate
        self.handle_type = ctypes.c_void_p
        self.handle: ctypes.c_uint64 = ctypes.c_void_p(0)
        self.grid_handle = grid_handle
        self.cardinality = ctypes.c_int(cardinality)
        self.py_grid = py_grid
        self._set_field_type()
        self._help_load_api()
        self._help_field_new()

    def __del__(self):
        self.help_delete()
        pass

    def _set_field_type(self):
        self.type_mapping = self.neon_gate.get_type_mapping(self.dtype)
        self.suffix = f'_{self.type_mapping["suffix"]}'
        self.Partition_type = getattr(neon.block.bPartition, f'bPartition{self.suffix}')


    def _help_load_api(self):
        # Importing new functions
        ## new_field
        lib_obj = self.neon_gate.lib

        # ---------------------------------------------------------------------
        self.api_new = getattr(lib_obj, f'bGrid_bField_new{self.suffix}')
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),
                                 self.handle_type,
                                 ctypes.c_int]
        self.api_new.restype = ctypes.c_int

        # ---------------------------------------------------------------------
        self.api_delete = getattr(lib_obj, f'bGrid_bField_delete{self.suffix}')
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.api_delete.restype = ctypes.c_int

        ## get_partition
        self.api_get_partition = getattr(lib_obj, f'bGrid_bField_get_partition{self.suffix}')
        self.api_get_partition.argtypes = [
            self.handle_type,
            ctypes.POINTER(self.Partition_type),  # the span object
            ctypes.c_int,  # the execution type
            ctypes.c_int,  # the device id
            ctypes.c_int,  # the data view
        ]
        self.api_get_partition.restype = ctypes.c_int

        # # size partition
        # self.neon.lib.bGrid_bField_partition_size.argtypes = [
        #     ctypes.POINTER(self.Partition_type)]
        # self.neon.lib.bGrid_bField_partition_size.restype = ctypes.c_int

        # field read
        self.api_read = getattr(lib_obj, f'bGrid_bField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,
                                  ctypes.POINTER(neon.Index_3d),
                                  ctypes.c_int]
        self.api_read.restype = self.type_mapping["ctype"]

        # field write
        self.api_write = getattr(lib_obj, f'bGrid_bField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,
                                   ctypes.POINTER(neon.Index_3d),
                                   ctypes.c_int,
                                   self.type_mapping["ctype"]]
        self.api_write.restype = ctypes.c_int

        # field update host data
        self.api_update_host = getattr(lib_obj, f'bGrid_bField_update_host_data{self.suffix}')
        self.api_update_host.argtypes = [self.handle_type,
                                         ctypes.c_int]
        self.api_update_host.restype = ctypes.c_int

        # field update device data
        self.api_update_device = getattr(lib_obj, f'bGrid_bField_update_device_data{self.suffix}')
        self.api_update_device.argtypes = [self.handle_type,
                                           ctypes.c_int]
        self.api_update_device.restype = ctypes.c_int

        # # export vti
        # self.api_export_vti = getattr(lib_obj, f'bGrid_bField_to_vti{self.suffix}')
        # self.api_export_vti.argtypes = [self.handle_type,
        #                                    ctypes.c_char_p,
        #                                    ctypes.c_char_p]
        # self.api_export_vti.restype = ctypes.c_int

    def _help_field_new(self):
        if self.handle == 0:
            raise Exception('bGrid: Invalid handle')

        res = self.api_new(ctypes.pointer(self.handle),
                           self.grid_handle,
                           self.cardinality)
        if res != 0:
            raise Exception('bGrid: Failed to initialize field')

    def help_delete(self):
        if self.handle == 0:
            return
        res = self.api_delete(ctypes.pointer(self.handle))
        if res != 0:
            raise Exception('Failed to delete field')

    def get_grid(self):
        return self.py_grid

    def get_partition(self,
                      execution: neon.Execution,
                      c: ctypes.c_int,
                      data_view: neon.DataView
                      ):
        if self.handle == 0:
            raise Exception('bField: Invalid handle')

        partition = self.Partition_type()

        res = self.api_get_partition(self.handle,
                                     partition,
                                     int(execution),
                                     c,
                                     int(data_view))
        if res != 0:
            raise Exception('Failed to get partition')

        # ccp_size = self.neon.lib.bGrid_bField_partition_size(partition)
        # ctypes_size = ctypes.sizeof(partition)
        #
        # if ccp_size != ctypes_size:
        #     raise Exception(f'Failed to get span: cpp_size {ccp_size} != ctypes_size {ctypes_size}')
        #
        # # print(f"Partition {partition}")
        return partition

    def read(self, idx: neon.Index_3d, cardinality: ctypes.c_int):
        return self.api_read(self.handle,
                             idx,
                             cardinality)

    def write(self, idx: neon.Index_3d, cardinality: ctypes.c_int, newValue):
        return self.api_write(self.handle,
                              idx,
                              cardinality,
                              self.type_mapping['ctype'](newValue))

    def update_host(self, streamSetId: ctypes.c_int):
        return self.api_update_host(self.handle,
                                    streamSetId)

    def update_device(self, streamSetId: ctypes.c_int):
        return self.api_update_device(self.handle,
                                      streamSetId)

    def export_vti(self, filename: str,
                   field_name: str = "field"):
        self.api_export_vti(self.handle, filename.encode('utf-8'), field_name.encode('utf-8'))

    def get_cardinality(self):
        return self.cardinality.value

    def get_type(self):
        return self.dtype

    def get_handle(self):
        return self.handle

    @property
    def type(self):
        return self.dtype
