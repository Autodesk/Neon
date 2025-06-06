#include "Neon/py/mGrid.h"
#include <nvtx3/nvToolsExt.h>
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"
extern "C" auto mGrid_new(
    void**                handle,
    void*                 backendPtr,
    const Neon::index_3d* dim,
    int32_t               num_levels,
    int**                 sparsity_pattern_vec,
    Neon::index_3d*       dim_vec,
    Neon::index_3d*       origin_vec,
    int                   numStencilPoints,
    int const*            stencilPointFlatArray)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);
    std::cout << "mGrid_new - BEGIN" << std::endl;
    std::cout << "mGrid_new - gridHandle " << handle << std::endl;
    std::cout << "mGrid_new - dim " << dim->to_string() << std::endl;

    Neon::init();

    using Grid = Neon::domain::mGrid;

    Neon::Backend* backend = reinterpret_cast<Neon::Backend*>(backendPtr);
    if (backend == nullptr) {
        std::cerr << "Invalid backend pointer" << std::endl;
        return -1;
    }

    std::vector<std::function<bool(const Neon::index_3d&)>> sparsity(num_levels);
    for (int i = 0; i < num_levels; i++) {
        Neon::index_3d const level_mask_dim = dim_vec[i];
        Neon::index_3d const level_mask_origin = origin_vec[i];
        int* const           level_sparsity = sparsity_pattern_vec[i];
        int const            dividend = 1 << i;
        // std::cout << "Level " << i << " dividend " << dividend << std::endl;
        // std::cout << "Pattern Dimension: " << level_mask_dim.to_string() << std::endl;
        // std::cout << "Pattern Origin: " << level_mask_origin.to_string() << std::endl;

        if (i == 1) {
            for (int z = 0; z < level_mask_dim.z; z++) {
                std::cout << "." << std::endl;
                for (int y = 0; y < level_mask_dim.y; y++) {
                    for (int x = 0; x < level_mask_dim.x; x++) {
                        int index = x * (level_mask_dim.y * level_mask_dim.z) + y * level_mask_dim.z + z;
                        std::cout << "YESS " << level_sparsity[index] << " x->" << x << "  y->" << y << "  z->" << z << std::endl;
                    }
                }
            }
        }

        sparsity[i] = [=](Neon::index_3d const& idx) {
            auto const scaled_idx = idx / dividend;
            auto const mask_idx = scaled_idx - level_mask_origin;

            if (mask_idx.x < 0 || mask_idx.y < 0 || mask_idx.z < 0) {
                if (i == 1) {
                    std::cout << "LINE 61 idx " << idx << " scaled_idx " << scaled_idx << " mask_idx " << mask_idx << std::endl;
                }
                return false;
            }
            if (mask_idx.x >= level_mask_dim.x || mask_idx.y >= level_mask_dim.y || mask_idx.z >= level_mask_dim.z) {
                if (i == 1) {
                    std::cout << "LINE 67 idx " << idx << " scaled_idx " << scaled_idx << " mask_idx " << mask_idx << " level_mask_dim " << level_mask_dim <<std::endl;
                }
                return false;
            }
            // int idx = i * (dim1 * dim2) + j * dim2 + k;
            // int32_t value = data[idx];
            // std::printf("arr[%d][%d][%d] = %d\n", i, j, k, value);
            if (i == 1)
                std::cout << "CHECK idx " << idx << " scaled_idx " << scaled_idx << " " << std::endl;
            int index = mask_idx.x * (level_mask_dim.y * level_mask_dim.z) +
                        mask_idx.y * level_mask_dim.z +
                        mask_idx.z;
            return level_sparsity[index] == 1;
        };
    }

    std::vector<Neon::index_3d> points(numStencilPoints);
    for (int sId = 0; sId < numStencilPoints; sId++) {
        points[sId].x = stencilPointFlatArray[sId * 3];
        points[sId].y = stencilPointFlatArray[sId * 3 + 1];
        points[sId].z = stencilPointFlatArray[sId * 3 + 2];
    }
    Neon::domain::Stencil stencil(points);
    auto                  gridPtr = new (std::nothrow)
        Grid(*backend,
             *dim,
             sparsity,
             stencil,
             Grid::Descriptor(num_levels));

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    *handle = (void*)gridPtr;
    std::cout << "grid_new - END" << std::endl;

    // g.ioDomainToVtk("");
    NEON_PY_PRINT_END(*handle);

    return 0;
}


extern "C" auto mGrid_delete(
    void** handle)
    -> int
{
    std::cout << "mGrid_delete - BEGIN" << std::endl;
    std::cout << "mGrid_delete - gridHandle " << handle << std::endl;

    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(*handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
    }
    *handle = nullptr;

    std::cout << "mGrid_delete - END" << std::endl;
    return 0;
}

extern "C" auto mGrid_get_dimensions(
    void*           gridHandle,
    Neon::index_3d* dim)
    -> int
{
    std::cout << "mGrid_get_dimension - BEGIN" << std::endl;
    std::cout << "mGrid_get_dimension - gridHandle " << gridHandle << std::endl;


    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: gridHandle is invalid " << std::endl;
        return -1;
    }

    auto dimension = gridPtr->getDimension();
    dim->x = dimension.x;
    dim->y = dimension.y;
    dim->z = dimension.z;

    std::cout << "mGrid_get_dimension - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

extern "C" auto mGrid_get_span(
    void*                      gridHandle,
    int32_t                    grid_level,
    Neon::domain::mGrid::Span* spanRes,
    int                        execution,
    int                        device,
    int                        data_view)
    -> int
{
    std::cout << "mGrid_get_span - BEGIN " << std::endl;
    // std::cout << "mGrid_get_span - gridHandle " << gridHandle << std::endl;
    // std::cout << "mGrid_get_span - grid_level " << grid_level << std::endl;
    // std::cout << "mGrid_get_span - execution " << execution << std::endl;
    // std::cout << "mGrid_get_span - device " << device << std::endl;
    // std::cout << "mGrid_get_span - data_view " << data_view << std::endl;
    // std::cout << "mGrid_get_span - Span size " << sizeof(*spanRes) << std::endl;

    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        if (!(grid_level < int(grid.getLevelCount()))) {
            std::cout << "grid_level out of range in mGrid_get_span LEVEL" << grid_level << " of " << grid.getLevelCount() << std::endl;
        }
        auto& gridSpan = grid(grid_level).getSpan(Neon::ExecutionUtils::fromInt(execution), device, Neon::DataViewUtil::fromInt(data_view));
        (*spanRes) = gridSpan;
        std::cout << "mGrid_get_span - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

template <typename T>
auto mGrid_mField_new(
    void** fieldHandle,
    void*  gridHandle,
    int    cardinality)
    -> int
{
    NEON_PY_PRINT_BEGIN(*fieldHandle);

    std::cout << "mGrid_mField_new - BEGIN" << std::endl;
    std::cout << "mGrid_mField_new - fieldHandle: " << fieldHandle << std::endl;
    std::cout << "mGrid_mField_new - gridHandle: " << gridHandle << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        Field  field = grid.newField<T, 0>("test", cardinality, 0, Neon::DataUse::HOST_DEVICE);
        Field* fieldPtr = new (std::nothrow) Field(field);
        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }

        // auto partition = fieldPtr->operator()(0).getPartition(Neon::Execution::device, 0, Neon::DataView::INTERNAL);
        // std::cout << "mGrid_mField_new - partition cardinality " << partition.cardinality() << std::endl;

        *fieldHandle = fieldPtr;
        NEON_PY_PRINT_END(*fieldHandle);

        return 0;
    }
    std::cout << "mGrid_mField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

DO_EXPORT(int8, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint8, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(bool, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int32, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint32, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int64, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint64, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(float32, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(float64, 3, mGrid_mField_new, int, void**, handle, void*, gridHandle, int, cardinality);

template <typename T>
auto mGrid_mField_delete(
    void** handle)
    -> int
{
    std::cout << "mGrid_mField_delete - BEGIN" << std::endl;
    std::cout << "mGrid_mField_delete - handle " << handle << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = (Field*)(*handle);

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Deallocation();
    }
    *handle = 0;
    std::cout << "mGrid_mField_delete - END" << std::endl;

    return 0;
}

DO_EXPORT(int8, 1, mGrid_mField_delete, int, void**, handle);
DO_EXPORT(uint8, 1, mGrid_mField_delete, int, void**, handle);
DO_EXPORT(bool, 1, mGrid_mField_delete, int, void**, handle);

DO_EXPORT(int32, 1, mGrid_mField_delete, int, void**, handle);
DO_EXPORT(uint32, 1, mGrid_mField_delete, int, void**, handle);

DO_EXPORT(int64, 1, mGrid_mField_delete, int, void**, handle);
DO_EXPORT(uint64, 1, mGrid_mField_delete, int, void**, handle);

DO_EXPORT(float32, 1, mGrid_mField_delete, int, void**, handle);
DO_EXPORT(float64, 1, mGrid_mField_delete, int, void**, handle);

template <typename T>
auto mGrid_mField_get_partition(
    void*                                                  field_handle,
    [[maybe_unused]] Neon::domain::mGrid::Partition<T, 0>* partitionPtr,
    int32_t                                                resolution_level,
    Neon::Execution                                        execution,
    int                                                    device,
    Neon::DataView                                         data_view)
    -> int
{
    NEON_PY_PRINT_BEGIN(field_handle);

    std::cout << "mGrid_mField_get_partition - BEGIN " << std::endl;
    std::cout << "mGrid_mField_get_partition - field_handle " << field_handle << std::endl;
    std::cout << "mGrid_mField_get_partition - execution " << Neon::ExecutionUtils::toString(execution) << std::endl;
    std::cout << "mGrid_mField_get_partition - resolution_level " << resolution_level << std::endl;
    std::cout << "mGrid_mField_get_partition - device " << device << std::endl;
    std::cout << "mGrid_mField_get_partition - data_view " << Neon::DataViewUtil::toString(data_view) << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = (Field*)field_handle;

    if (fieldPtr != nullptr) {
        const auto& descriptor = fieldPtr->getDescriptor();

        // check to make sure that the given field level is within bounds. The first clause is to allow a cast in the second clause.
        if (descriptor.getDepth() < 0 || resolution_level >= descriptor.getDepth()) {
            std::cout << "field index out of bounds" << std::endl;
            return -1;
        }
        auto p = (*fieldPtr)(resolution_level).getPartition(execution, device, data_view);
        std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;

        std::cout << "mGrid_mField_get_partition - END" << std::endl;
        NEON_PY_PRINT_END(field_handle);

        return 0;
    }
    return -1;
}

DO_EXPORT(int8, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<int8, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint8, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<uint8, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(bool, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<bool, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(int32, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<int32, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint32, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<uint32, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(int64, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<int64, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint64, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<uint64, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(float32, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<float32, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(float64, 6, mGrid_mField_get_partition, int, void*, field_handle, decltype(Neon::domain::mGrid::Partition<float64, 0>())*, partitionPtr, int, resolution_level, Neon::Execution, execution, int, device, Neon::DataView, data_view);

auto mGrid_span_size(
    Neon::domain::mGrid::Span* spanRes)
    -> int
{
    return sizeof(*spanRes);
}

auto mGrid_mField_partition_size(
    Neon::domain::mGrid::Partition<int, 0>* partitionPtr)
    -> int
{
    return sizeof(*partitionPtr);
}


// auto mGrid_get_properties(/* TODOMATT verify what the return of this method should be */
//                           void*                 gridHandle,
//                           uint64_t              grid_level,
//                           const Neon::index_3d* idx)
//     -> int
// {
//     std::cout << "mGrid_get_properties begin" << std::endl;
//
//     using Grid = Neon::domain::mGrid;
//     Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
//     if (grid_level >= gridPtr->getGridCount()) {
//         std::cout << "grid_level out of range in mGrid_get_properties" << std::endl;
//     }
//
//     int returnValue = int((*gridPtr)(grid_level).getProperties(*idx).getDataView());
//     std::cout << "mGrid_get_properties end" << std::endl;
//
//     return returnValue;
// }
//
extern "C" auto mGrid_is_inside_domain(
    void*                 gridHandle,
    uint64_t              grid_level,
    const Neon::index_3d* idx)
    -> bool
{
    std::cout << "mGrid_is_inside_domain begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    bool returnValue = gridPtr->isInsideDomain(*idx, grid_level);

    std::cout << "mGrid_is_inside_domain end" << std::endl;

    return returnValue;
}

template <typename T>
auto mGrid_mField_read(
    void*                 fieldHandle,
    int32_t               resolution_level,
    const Neon::index_3d* idx,
    const int             cardinality)
    -> T
{
    std::cout << "mGrid_mField_read begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(*idx, cardinality, resolution_level);

    std::cout << "mGrid_mField_read end" << std::endl;

    return returnValue;
}


DO_EXPORT(int8, 4, mGrid_mField_read, int8, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint8, 4, mGrid_mField_read, uint8, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(bool, 4, mGrid_mField_read, bool, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int32, 4, mGrid_mField_read, int32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint32, 4, mGrid_mField_read, uint32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int64, 4, mGrid_mField_read, int64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint64, 4, mGrid_mField_read, uint64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(float32, 4, mGrid_mField_read, float32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(float64, 4, mGrid_mField_read, float64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality);


template <typename T>
auto mGrid_mField_write(
    void*                 fieldHandle,
    int32_t               resolution_level,
    const Neon::index_3d* idx,
    const int             cardinality,
    T                     newValue)
    -> int
{
    std::cout << "mGrid_mField_write begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->getReference(*idx, cardinality, resolution_level) = newValue;

    std::cout << "mGrid_mField_write end" << std::endl;
    return 0;
}

DO_EXPORT(int8, 5, mGrid_mField_write, int8, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, int8, newValue);
DO_EXPORT(uint8, 5, mGrid_mField_write, uint8, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, uint8, newValue);
DO_EXPORT(bool, 5, mGrid_mField_write, bool, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, bool, newValue);

DO_EXPORT(int32, 5, mGrid_mField_write, int32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, int32, newValue);
DO_EXPORT(uint32, 5, mGrid_mField_write, uint32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, uint32, newValue);

DO_EXPORT(int64, 5, mGrid_mField_write, int64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, int64, newValue);
DO_EXPORT(uint64, 5, mGrid_mField_write, uint64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, uint64, newValue);

DO_EXPORT(float32, 5, mGrid_mField_write, float32, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, float32, newValue);
DO_EXPORT(float64, 5, mGrid_mField_write, float64, void*, fieldHandle, int, resolution_level, const Neon::index_3d*, idx, const int, cardinality, float64, newValue);

template <typename T>
auto mGrid_mField_update_host_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
    std::cout << "mGrid_mField_update_host_data begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateHostData(streamSetId);

    std::cout << "mGrid_mField_update_host_data end" << std::endl;
    return 0;
}

DO_EXPORT(int8, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, mGrid_mField_update_host_data, int, void*, fieldHandle, int, streamSetId);

template <typename T>
auto mGrid_mField_update_device_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
    std::cout << "mGrid_mField_update_device_data begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateDeviceData(streamSetId);

    std::cout << "mGrid_mField_update_device_data end" << std::endl;
    return 0;
}

DO_EXPORT(int8, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, mGrid_mField_update_device_data, int, void*, fieldHandle, int, streamSetId);


template <typename T>
auto mGrid_mField_to_vti(
    void*       fieldHandle,
    const char* fname,
    const char* fieldName,
    bool        outputLevels,
    bool        outputBlockID,
    bool        outputVoxelID,
    bool        filterOverlaps)
    -> int
{

#ifdef NEON_USE_NVTX
    nvtxRangePush("mGrid_mField_to_vti");
#endif

    NEON_PY_PRINT_BEGIN(fieldHandle);

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    std::cout << "mGrid_mField_to_vti - " << fname << " - " << fieldName << std::endl;
    fieldPtr->ioToVtk(fname, outputLevels, outputBlockID, outputVoxelID, filterOverlaps);
    //                      bool               includeDomain = false,
    //                      Neon::IoFileType   ioFileType = Neon::IoFileType::ASCII,
    //                      bool               isNodeSpace = false
    // fieldPtr->updateHostData(streamSetId);

#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    NEON_PY_PRINT_END(fieldHandle);

    return 0;
}

DO_EXPORT(int8, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldNamebool, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);
DO_EXPORT(uint8, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);
DO_EXPORT(bool, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);

DO_EXPORT(int32, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);
DO_EXPORT(uint32, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);

DO_EXPORT(int64, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);
DO_EXPORT(uint64, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);

DO_EXPORT(float32, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);
DO_EXPORT(float64, 7, mGrid_mField_to_vti, int, void*, fieldHandle, const char*, fname, const char*, fieldName, bool, outputLevels, bool, outputBlockID, bool, outputVoxelID, bool, filterOverlaps);


template <typename T>
auto mGrid_mField_to_vti_debug(
    void*       fieldHandle,
    const char* fname,
    const char* fieldName)
    -> int
{
#ifdef NEON_USE_NVTX
    nvtxRangePush("mGrid_mField_to_vti");
#endif

    NEON_PY_PRINT_BEGIN(fieldHandle);

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    std::cout << "mGrid_mField_to_vti - " << fname << " - " << fieldName << std::endl;
    fieldPtr->ioToVtk(fname, true, true, true, false);
    //                      bool               includeDomain = false,
    //                      Neon::IoFileType   ioFileType = Neon::IoFileType::ASCII,
    //                      bool               isNodeSpace = false
    // fieldPtr->updateHostData(streamSetId);

#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    NEON_PY_PRINT_END(fieldHandle);

    return 0;
}

DO_EXPORT(int8, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);
DO_EXPORT(uint8, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);
DO_EXPORT(bool, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);

DO_EXPORT(int32, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);
DO_EXPORT(uint32, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);

DO_EXPORT(int64, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);
DO_EXPORT(uint64, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);

DO_EXPORT(float32, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);
DO_EXPORT(float64, 3, mGrid_mField_to_vti_debug, int, void*, fieldHandle, const char*, fname, const char*, fieldName);

extern "C" auto mGrid_mField_mPartition_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void
{
    Neon::domain::mGrid::Partition<int, 0>::getOffsets(offsets, length);
}
