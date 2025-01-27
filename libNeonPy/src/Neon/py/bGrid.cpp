#include <nvtx3/nvToolsExt.h>
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"

extern "C" auto bGrid_new(
    void**                handle,
    void*                 backendPtr,
    const Neon::index_3d* dim,
    [[maybe_unused]] int*                  sparsity_pattern,
    int                   numStencilPoints,
    int const*            stencilPointFlatArray)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);

    Neon::init();

    using Grid = Neon::bGrid;

    // Backend
    Neon::Backend* backend = reinterpret_cast<Neon::Backend*>(backendPtr);
    if (backend == nullptr) {
        std::cerr << "Invalid backend pointer" << std::endl;
        return -1;
    }

    // Stencil
    std::vector<Neon::index_3d> points(numStencilPoints);
    for (int sId = 0; sId < numStencilPoints; sId++) {
        points[sId].x = stencilPointFlatArray[sId * 3];
        points[sId].y = stencilPointFlatArray[sId * 3 + 1];
        points[sId].z = stencilPointFlatArray[sId * 3 + 2];
    }
    Neon::domain::Stencil stencil(points);

    // Grid
    Grid g(
        *backend,
        *dim,
        [=](Neon::index_3d const& /*idx*/) {
            return true;
            //return sparsity_pattern[idx.x * (dim->x * dim->y) + idx.y * dim->z + idx.z];
        },
        stencil);
    auto gridPtr = new (std::nothrow) Grid(g);
    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocate grid " << std::endl;
        return -1;
    } else {
        AllocationCounter::Allocation();
    }
    // Returned values
    *handle = (void*)gridPtr;
    std::cout << "grid_new - END" << std::endl;
    // g.ioDomainToVtk("")
    NEON_PY_PRINT_END(*handle);

    return 0;
}

extern "C" auto bGrid_delete(
    void** handle)
    -> int
{
    std::cout << "bGrid_delete - BEGIN" << std::endl;
    std::cout << "bGrid_delete - gridHandle " << handle << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(*handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
        AllocationCounter::Deallocation();
    }
    *handle = 0;
    std::cout << "bGrid_delete - END" << std::endl;
    return 0;
}

extern "C" auto bGrid_get_dimensions(
    void*           gridHandle,
    Neon::index_3d* dim)
    -> int
{
    std::cout << "bGrid_get_dimension - BEGIN" << std::endl;
    std::cout << "bGrid_get_dimension - gridHandle " << gridHandle << std::endl;


    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: gridHandle is invalid " << std::endl;
        return -1;
    }

    auto dimension = gridPtr->getDimension();
    dim->x = dimension.x;
    dim->y = dimension.y;
    dim->z = dimension.z;

    std::cout << "bGrid_get_dimension - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

extern "C" auto bGrid_get_span(
    void*              gridHandle,
    Neon::bGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int
{
    std::cout << "bGrid_get_span - BEGIN " << std::endl;
    std::cout << "bGrid_get_span - gridHandle " << gridHandle << std::endl;
    std::cout << "bGrid_get_span - execution " << execution << std::endl;
    std::cout << "bGrid_get_span - device " << device << std::endl;
    std::cout << "bGrid_get_span - data_view " << data_view << std::endl;
    std::cout << "bGrid_get_span - Span size " << sizeof(*spanRes) << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        auto& gridSpan = grid.getSpan(Neon::ExecutionUtils::fromInt(execution),
                                      device,
                                      Neon::DataViewUtil::fromInt(data_view));
        (*spanRes) = gridSpan;
        std::cout << "bGrid_get_span - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

template <typename T>
auto bGrid_bField_new(
    void** fieldHandle,
    void*  gridHandle,
    int    cardinality)
    -> int
{
    std::cout << "bGrid_bField_new - BEGIN" << std::endl;
    std::cout << "bGrid_bField_new - gridHandle " << gridHandle << std::endl;
    std::cout << "bGrid_bField_new - fieldHandle " << fieldHandle << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        using Field = Grid::Field<T, 0>;
        Field field = grid.newField<T, 0>("test", cardinality, 0, Neon::DataUse::HOST_DEVICE);
        // std::cout << field.toString() << std::endl;
        Field* fieldPtr = new (std::nothrow) Field(field);
        AllocationCounter::Allocation();

        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }

        *fieldHandle = (void*)fieldPtr;
        std::cout << "bGrid_bField_new - END " << *fieldHandle << std::endl;

        return 0;
    }
    std::cout << "bGrid_bField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

DO_EXPORT(int8, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint8, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(bool, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int32, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint32, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int64, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint64, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(float32, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(float64, 3, bGrid_bField_new, int, void**, handle, void*, gridHandle, int, cardinality);

template <typename T>
auto bGrid_bField_delete(
    void** handle)
    -> int
{
    std::cout << "bGrid_bField_delete - BEGIN" << std::endl;
    std::cout << "bGrid_bField_delete - handle " << handle << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    auto fieldPtr = (Field*)(*handle);

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Deallocation();
    }

    *handle = 0;
    std::cout << "bGrid_bField_delete - END" << std::endl;

    return 0;
}

DO_EXPORT(int8, 1, bGrid_bField_delete, int, void**, handle);
DO_EXPORT(uint8, 1, bGrid_bField_delete, int, void**, handle);
DO_EXPORT(bool, 1, bGrid_bField_delete, int, void**, handle);

DO_EXPORT(int32, 1, bGrid_bField_delete, int, void**, handle);
DO_EXPORT(uint32, 1, bGrid_bField_delete, int, void**, handle);

DO_EXPORT(int64, 1, bGrid_bField_delete, int, void**, handle);
DO_EXPORT(uint64, 1, bGrid_bField_delete, int, void**, handle);

DO_EXPORT(float32, 1, bGrid_bField_delete, int, void**, handle);
DO_EXPORT(float64, 1, bGrid_bField_delete, int, void**, handle);

template <typename T>
auto bGrid_bField_get_partition(
    void*                                          field_handle,
    [[maybe_unused]] Neon::bGrid::Partition<T, 0>* partitionPtr,
    Neon::Execution                                execution,
    int                                            device,
    Neon::DataView                                 data_view)
    -> int
{
    NEON_PY_PRINT_BEGIN(field_handle);

    // std::cout << "bGrid_bField_get_partition - BEGIN " << std::endl;
    // std::cout << "bGrid_bField_get_partition - field_handle " << field_handle << std::endl;
    // std::cout << "bGrid_bField_get_partition - execution " << Neon::ExecutionUtils::toString(execution) << std::endl;
    // std::cout << "bGrid_bField_get_partition - device " << device << std::endl;
    // std::cout << "bGrid_bField_get_partition - data_view " << Neon::DataViewUtil::toString(data_view) << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    auto fieldPtr = (Field*)field_handle;

    if (fieldPtr != nullptr) {
        auto p = fieldPtr->getPartition(execution,
                                        device,
                                        data_view);
        std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;
        NEON_PY_PRINT_END(field_handle);

        return 0;
    }
    NEON_PY_PRINT_END(field_handle);
    return -1;
}

DO_EXPORT(int8, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<int8, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint8, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<uint8, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(bool, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<bool, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(int32, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<int32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint32, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<uint32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(int64, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<int64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint64, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<uint64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(float32, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<float32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(float64, 5, bGrid_bField_get_partition, int, void*, field_handle, decltype(Neon::bGrid::Partition<float64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);


extern "C" auto bGrid_span_size(
    Neon::bGrid::Span* spanRes)
    -> int
{
    return sizeof(*spanRes);
}

extern "C" auto bGrid_bField_partition_size(
    Neon::bGrid::Partition<int, 0>* partitionPtr)
    -> int
{
    return sizeof(*partitionPtr);
}

extern "C" auto bGrid_get_properties(/* TODOMATT verify what the return of this method should be */
                                     uint64_t&             gridHandle,
                                     const Neon::index_3d* idx)
    -> int
{
    std::cout << "bGrid_get_properties begin" << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    int   returnValue = int(gridPtr->getProperties(*idx).getDataView());
    std::cout << "bGrid_get_properties end" << std::endl;

    return returnValue;
}

extern "C" auto bGrid_is_inside_domain(
    void*                       gridHandle,
    const Neon::index_3d* const idx)
    -> bool
{
    std::cout << "bGrid_is_inside_domain begin" << std::endl;

    using Grid = Neon::bGrid;
    auto* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    bool  returnValue = gridPtr->isInsideDomain(*idx);

    std::cout << "bGrid_is_inside_domain end" << std::endl;


    return returnValue;
}

template <typename T>
auto bGrid_bField_read(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    const int             cardinality)
    -> int
{
    std::cout << "bGrid_bField_read begin" << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(*idx, cardinality);

    std::cout << "bGrid_bField_read end" << std::endl;

    return returnValue;
}

DO_EXPORT(int8, 3, bGrid_bField_read, int8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint8, 3, bGrid_bField_read, uint8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(bool, 3, bGrid_bField_read, uint8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int32, 3, bGrid_bField_read, int32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint32, 3, bGrid_bField_read, uint32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int64, 3, bGrid_bField_read, int64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint64, 3, bGrid_bField_read, uint64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(float32, 3, bGrid_bField_read, float32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(float64, 3, bGrid_bField_read, float64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

template <typename T>
auto bGrid_bField_write(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    const int             cardinality,
    int                   newValue)
    -> int
{
    std::cout << "bGrid_bField_write begin" << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    std::cout <<fieldPtr->toString() << std::endl;

    fieldPtr->getReference(*idx, cardinality) = newValue;

    std::cout << "bGrid_bField_write end" << std::endl;
    return 0;
}


DO_EXPORT(int8, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int8, newValue);
DO_EXPORT(uint8, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint8, newValue);
DO_EXPORT(bool, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint8, newValue);

DO_EXPORT(int32, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int32, newValue);
DO_EXPORT(uint32, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint32, newValue);

DO_EXPORT(int64, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int64, newValue);
DO_EXPORT(uint64, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint64, newValue);

DO_EXPORT(float32, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, float32, newValue);
DO_EXPORT(float64, 4, bGrid_bField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, float64, newValue);

template <typename T>
auto bGrid_bField_update_host_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
#ifdef NEON_USE_NVTX
    nvtxRangePush("bGrid_bField_update_host_data");
#endif

    NEON_PY_PRINT_BEGIN(fieldHandle);

    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateHostData(streamSetId);

#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    NEON_PY_PRINT_END(fieldHandle);
    return 0;
}


DO_EXPORT(int8, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, bGrid_bField_update_host_data, int, void*, fieldHandle, int, streamSetId);

template <typename T>
auto bGrid_bField_update_device_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
    NEON_PY_PRINT_BEGIN(fieldHandle);

#ifdef NEON_USE_NVTX
    nvtxRangePush("bGrid_bField_update_device_data");
#endif
    using Grid = Neon::bGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateDeviceData(streamSetId);

#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    NEON_PY_PRINT_END(fieldHandle);
    return 0;
}

DO_EXPORT(int8, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, bGrid_bField_update_device_data, int, void*, fieldHandle, int, streamSetId);


extern "C" auto bGrid_bSpan_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void
{
    Neon::domain::details::bGrid::bSpan<Neon::domain::details::bGrid::BlockDefault>::getOffsets(offsets, length);
}

extern "C" auto bGrid_bField_bPartition_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void
{
    Neon::domain::details::bGrid::bPartition<int, 0, Neon::domain::details::bGrid::BlockDefault>::getOffsets(offsets, length);
}
