#include "Neon/py/dGrid.h"
#include "Neon/domain/Grids.h"
#include "Neon/set/Backend.h"
#include "Neon/py/AllocationCounter.h"
#include <nvtx3/nvToolsExt.h>

auto dGrid_new(
    uint64_t& handle,
    uint64_t& backendPtr,
    const Neon::index_3d* dim,
    int* sparsity_pattern)
    -> int
{
    std::cout << "dGrid_new - BEGIN" << std::endl;
    std::cout << "dGrid_new - gridHandle " << handle << std::endl;

    Neon::init();

    using Grid = Neon::dGrid;

    Neon::Backend* backend = reinterpret_cast<Neon::Backend*>(backendPtr);
    if (backend == nullptr) {
        std::cerr << "Invalid backend pointer" << std::endl;
        return -1;
    }

    // Neon::index_3d dim{x,y,z};
    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    Grid                  g(*backend, *dim, [=](Neon::index_3d const& idx) { return sparsity_pattern[idx.x * (dim->x * dim->y) + idx.y * dim->z + idx.z ]; }, d3q19);
    auto                  gridPtr = new (std::nothrow) Grid(g);
    AllocationCounter::Allocation();

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    handle = reinterpret_cast<uint64_t>(gridPtr);
    std::cout << "grid_new - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

auto dGrid_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "dGrid_delete - gridHandle " << handle << std::endl << std::flush;
    std::cout << "dGrid_delete - BEGIN" << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
        AllocationCounter::Deallocation();
    }
    handle = 0;
    std::cout << "dGrid_delete - END" << std::endl;
    return 0;
}

extern "C" auto dGrid_get_dimensions(
    uint64_t& gridHandle,
    Neon::index_3d* dim)
    -> int
{
    std::cout << "dGrid_get_dimension - BEGIN" << std::endl;
    std::cout << "dGrid_get_dimension - gridHandle " << gridHandle << std::endl;


    using Grid = Neon::dGrid;    
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: gridHandle is invalid " << std::endl;
        return -1;
    }

    auto dimension = gridPtr->getDimension();
    dim->x = dimension.x;
    dim->y = dimension.y;
    dim->z = dimension.z;

    std::cout << "dGrid_get_dimension - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

auto dGrid_get_span(
    uint64_t&          gridHandle,
    Neon::dGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int
{
    std::cout << "dGrid_get_span - BEGIN " << std::endl;
    std::cout << "dGrid_get_span - gridHandle " << gridHandle << std::endl;
    std::cout << "dGrid_get_span - execution " << execution << std::endl;
    std::cout << "dGrid_get_span - device " << device << std::endl;
    std::cout << "dGrid_get_span - data_view " << data_view << std::endl;
    std::cout << "dGrid_get_span - Span size " << sizeof(*spanRes) << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)gridHandle;
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        auto& gridSpan = grid.getSpan(Neon::ExecutionUtils::fromInt(execution),
                                      device,
                                      Neon::DataViewUtil::fromInt(data_view));
        (*spanRes) = gridSpan;
        std::cout << "dGrid_get_span - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

auto dGrid_dField_new(
    uint64_t& handle,
    uint64_t& gridHandle,
    int cardinality)
    -> int
{
    std::cout << "dGrid_dField_new - BEGIN" << std::endl;
    std::cout << "dGrid_dField_new - gridHandle " << gridHandle << std::endl;
    std::cout << "dGrid_dField_new - handle " << handle << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)gridHandle;
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        using Field = Grid::Field<int, 0>;
        Field field = grid.newField<int, 0>("test", cardinality, 0, Neon::DataUse::HOST_DEVICE);
        std::cout << field.toString() << std::endl;
        Field* fieldPtr = new(std::nothrow) Field(field);
        AllocationCounter::Allocation();

        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }
        handle = (uint64_t)fieldPtr;
        std::cout << "dGrid_dField_new - END " << handle << std::endl;

        return 0;
    }
    std::cout << "dGrid_dField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

auto dGrid_dField_get_partition(
    uint64_t&                                        field_handle,
    [[maybe_unused]] Neon::dGrid::Partition<int, 0>* partitionPtr,
    Neon::Execution                                  execution,
    int                                              device,
    Neon::DataView                                   data_view)
    -> int
{

    std::cout << "dGrid_dField_get_partition - BEGIN " << std::endl;
    std::cout << "dGrid_dField_get_partition - field_handle " << field_handle << std::endl;
    std::cout << "dGrid_dField_get_partition - execution " << Neon::ExecutionUtils::toString(execution) << std::endl;
    std::cout << "dGrid_dField_get_partition - device " << device << std::endl;
    std::cout << "dGrid_dField_get_partition - data_view " << Neon::DataViewUtil::toString(data_view) << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 0>;

    Field* fieldPtr = (Field*)field_handle;
    std::cout << fieldPtr->toString() << std::endl;

    if (fieldPtr != nullptr) {
        auto p = fieldPtr->getPartition(execution,
                                        device,
                                        data_view);
        std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;
        std::cout << "dGrid_dField_get_partition\n"
            << partitionPtr->to_string();

        std::cout << "dGrid_dField_get_partition - END" << std::endl;

        return 0;
    }
    return -1;
}

auto dGrid_dField_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "dGrid_dField_delete - BEGIN" << std::endl;
    std::cout << "dGrid_dField_delete - handle " << handle << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = (Field*)handle;

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Deallocation();
    }

    handle = 0;
    std::cout << "dGrid_dField_delete - END" << std::endl;

    return 0;
}

auto dGrid_span_size(
    Neon::dGrid::Span* spanRes)
    -> int
{
    return sizeof(*spanRes);
}

auto dGrid_dField_partition_size(
    Neon::dGrid::Partition<int, 0>* partitionPtr)
    -> int
{
    return sizeof(*partitionPtr);
}

auto dGrid_get_properties( /* TODOMATT verify what the return of this method should be */
    uint64_t& gridHandle,
    const Neon::index_3d* const idx) 
    -> int
{
    std::cout << "dGrid_get_properties begin" << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    int result = static_cast<int>(gridPtr->getProperties(*idx).getDataView());
    std::cout << "dGrid_get_properties end" << std::endl;

    return result;
}

auto dGrid_is_inside_domain(
    uint64_t& gridHandle,
    const Neon::index_3d* const idx)
    -> bool
{
    std::cout << "dGrid_is_inside_domain begin" << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    
    bool returnValue = gridPtr->isInsideDomain(*idx);

    std::cout << "dGrid_is_inside_domain end" << std::endl;


    return returnValue;
}

auto dGrid_dField_read(
    uint64_t& fieldHandle,
    const Neon::index_3d* idx,
    const int cardinality)
    -> int
{
    std::cout << "dGrid_dField_read begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(*idx, cardinality);
    
    std::cout << "dGrid_dField_read end" << std::endl;

    return returnValue;
}

auto dGrid_dField_write(
    uint64_t& fieldHandle,
    const Neon::index_3d* idx,
    int cardinality,
    int newValue)
    -> int
{
    std::cout << "dGrid_dField_write begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->getReference(*idx, cardinality) = newValue;
    
    std::cout << "dGrid_dField_write end" << std::endl;
    return 0;
}

auto dGrid_dField_update_host_data(
    uint64_t& fieldHandle,
    int       streamSetId)
    -> int
{
#ifdef NEON_USE_NVTX
    nvtxRangePush("dGrid_dField_update_host_data");
#endif

    std::cout << "dGrid_dField_update_host_data begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateHostData(streamSetId);

    std::cout << "dGrid_dField_update_host_data end" << std::endl;
#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    return 0;
}

auto dGrid_dField_update_device_data(
    uint64_t& fieldHandle,
    int       streamSetId)
    -> int
{
#ifdef NEON_USE_NVTX
    nvtxRangePush("dGrid_dField_update_host_data");
#endif
    std::cout << "dGrid_dField_update_device_data begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = (Field*)fieldHandle;

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateDeviceData(streamSetId);

    std::cout << "dGrid_dField_update_device_data end" << std::endl;
#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif
    return 0;
}

extern "C" auto dGrid_dSpan_get_member_field_offsets(size_t* offsets,
                                                     size_t* length)
    -> void
{
    Neon::domain::details::dGrid::dSpan::getOffsets(offsets, length);
}

extern "C" auto dGrid_dField_dPartition_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void
{
    Neon::domain::details::dGrid::dPartition<int,0>::getOffsets(offsets, length);
}
