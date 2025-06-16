import ctypes
import neon
import neon.multires.mPartition


class mField(object):
    def __init__(self,
                 neon_gate: neon.Gate,
                 grid_handle: ctypes.c_void_p,
                 cardinality: ctypes.c_int,
                 dtype,
                 py_grid,
                 ):
        self.dtype = dtype
        if grid_handle == 0:
            raise Exception('mField: Invalid handle')

        self.neon_gate: neon.Gate = neon_gate
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

    def _set_field_type(self):
        self.type_mapping = self.neon_gate.get_type_mapping(self.dtype)
        self.suffix = f'_{self.type_mapping["suffix"]}'
        self.Partition_type = getattr(neon.multires.mPartition, f'mPartition{self.suffix}')

    def _help_load_api(self):
        # Importing new functions
        ## new_field
        lib_obj = self.neon_gate.lib

        # ---------------------------------------------------------------------
        self.api_new = getattr(lib_obj, f'mGrid_mField_new{self.suffix}')
        self.api_new.argtypes = [ctypes.POINTER(self.handle_type),
                                 self.handle_type,
                                 ctypes.c_int]
        self.api_new.restype = ctypes.c_int

        # ---------------------------------------------------------------------
        self.api_delete = getattr(lib_obj, f'mGrid_mField_delete{self.suffix}')
        self.api_delete.argtypes = [ctypes.POINTER(self.handle_type)]
        self.api_delete.restype = ctypes.c_int

        ## get_partition
        self.api_get_partition = getattr(lib_obj, f'mGrid_mField_get_partition{self.suffix}')
        self.api_get_partition.argtypes = [
            self.handle_type,
            ctypes.POINTER(self.Partition_type),  # the span object
            ctypes.c_int,  # resolution level
            neon.Execution,  # the execution type
            ctypes.c_int,  # the device id
            neon.DataView,  # the data view
        ]
        self.api_get_partition.restype = ctypes.c_int

        # # size partition
        # self.neon.lib.mGrid_mField_partition_size.argtypes = [
        #     ctypes.POINTER(self.Partition_type)]
        # self.neon.lib.mGrid_mField_partition_size.restype = ctypes.c_int

        # field read
        self.api_read = getattr(lib_obj, f'mGrid_mField_read{self.suffix}')
        self.api_read.argtypes = [self.handle_type,
                                  ctypes.c_int32,  # resolution level
                                  ctypes.POINTER(neon.Index_3d),
                                  ctypes.c_int32]
        self.api_read.restype = self.type_mapping["ctype"]

        # field write
        self.api_write = getattr(lib_obj, f'mGrid_mField_write{self.suffix}')
        self.api_write.argtypes = [self.handle_type,
                                   ctypes.c_int32,  # resolution level
                                   ctypes.POINTER(neon.Index_3d),
                                   ctypes.c_int32,
                                   self.type_mapping["ctype"]]
        self.api_write.restype = ctypes.c_int

        # field update host data
        self.api_update_host = getattr(lib_obj, f'mGrid_mField_update_host_data{self.suffix}')
        self.api_update_host.argtypes = [self.handle_type,
                                         ctypes.c_int32]
        self.api_update_host.restype = ctypes.c_int32

        # field update device data
        self.api_update_device = getattr(lib_obj, f'mGrid_mField_update_device_data{self.suffix}')
        self.api_update_device.argtypes = [self.handle_type,
                                           ctypes.c_int32]
        self.api_update_device.restype = ctypes.c_int32

        # export vti
        self.api_export_vti = getattr(lib_obj, f'mGrid_mField_to_vti{self.suffix}')
        self.api_export_vti.argtypes = [self.handle_type,
                                        ctypes.c_char_p,
                                        ctypes.c_char_p,
                                        ctypes.c_bool,
                                        ctypes.c_bool,
                                        ctypes.c_bool
                                        ]  # field name
        self.api_export_vti.restype = ctypes.c_int32

        # export vti debug
        self.api_export_vti_debug = getattr(lib_obj, f'mGrid_mField_to_vti_debug{self.suffix}')
        self.api_export_vti.argtypes = [self.handle_type,
                                        ctypes.c_char_p,
                                        ctypes.c_char_p]
        self.api_export_vti.restype = ctypes.c_int32

        # field update host data
        self.api_fill = getattr(lib_obj, f'mGrid_mField_fill{self.suffix}')
        self.api_fill.argtypes = [self.handle_type,
                                  ctypes.c_int32,
                                  self.type_mapping["ctype"],
                                  ctypes.c_int]
        self.api_fill.restype = ctypes.c_int

        # self.api_fill = getattr(lib_obj, f'dGrid_dField_fill{self.suffix}')
        # self.api_fill.argtypes = [self.handle_type,
        #                           self.type_mapping["ctype"],
        #                           ctypes.c_int]
        # self.api_fill.restype = ctypes.c_int

        # field update host data
        self.api_copy = getattr(lib_obj, f'mGrid_mField_copy{self.suffix}')
        self.api_copy.argtypes = [self.handle_type,
                                  self.handle_type,
                                  ctypes.c_int,
                                  ctypes.c_int]
        self.api_copy.restype = ctypes.c_int

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
                      level: ctypes.c_int,
                      execution: neon.Execution,
                      device_id: ctypes.c_int,
                      data_view: neon.DataView
                      ):
        if self.handle == 0:
            raise Exception('bField: Invalid handle')

        partition = self.Partition_type()

        res = self.api_get_partition(self.handle,
                                     partition,
                                     level,
                                     execution,
                                     device_id,
                                     data_view)
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

    def read(self,
             level: ctypes.c_int,
             idx: neon.Index_3d,
             cardinality: ctypes.c_int):
        return self.api_read(self.handle,
                             level,
                             idx,
                             cardinality)

    def write(self,
              level: ctypes.c_int,
              idx: neon.Index_3d,
              cardinality: ctypes.c_int,
              newValue):
        return self.api_write(self.handle,
                              level,
                              idx,
                              cardinality,
                              self.type_mapping['ctype'](newValue))

    def update_host(self,
                    stream: ctypes.c_int):
        return self.api_update_host(self.handle,
                                    stream)

    def update_device(self,
                      stream: ctypes.c_int):
        return self.api_update_device(self.handle,
                                      stream)

    def export_vti(self, filename: str,
                   field_name: str = "field",
                   outputLevels: bool = True,
                   outputBlockID: bool = True,
                   outputVoxelID: bool = True,
                   filterOverlaps: bool = True):
        self.api_export_vti(self.handle, filename.encode('utf-8'), field_name.encode('utf-8'),
                            outputLevels,
                            outputBlockID,
                            outputVoxelID,
                            filterOverlaps)

    def get_cardinality(self):
        return self.cardinality.value

    def get_type(self):
        return self.dtype

    def get_handle(self):
        return self.handle

    def copy_from_run(self, level, src_field, stream_idx):
        self.api_copy(self.handle, src_field.handle, level, stream_idx)

    def fill_run(self, level, value, stream_idx):
        value = self.type_mapping['ctype'](value)
        # print(f"fill_run: value type: {type(value)}, expected ctype: {self.type_mapping['ctype']}")
        # print(f"fill_run: stream_idx type: {type(stream_idx)}, expected ctype: {ctypes.c_int}")

        self.api_fill(self.get_handle(),
                      level,
                      value.value,
                      stream_idx
                      )

    def zero_run(self, level, stream_idx):
        # print(f"zero_run: stream_idx type: {type(stream_idx)}, expected ctype: {ctypes.c_int}")
        self.fill_run(value=self.dtype(0), level=level, stream_idx=stream_idx)

    @property
    def type(self):
        return self.dtype
