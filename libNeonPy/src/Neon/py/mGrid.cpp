#include "Neon/py/mGrid.h"
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"

auto mGrid_new(
    uint64_t& handle,
    uint64_t& backendPtr,
    const Neon::index_3d* dim,
    uint32_t& depth)
    -> int
{
    std::cout << "mGrid_new - BEGIN" << std::endl;
    std::cout << "mGrid_new - gridHandle " << handle << std::endl;

    Neon::init();

    using Grid = Neon::domain::mGrid;

    Neon::Backend* backend = reinterpret_cast<Neon::Backend*>(backendPtr);
    if (backend == nullptr) {
        std::cerr << "Invalid backend pointer" << std::endl;
        return -1;
    }

    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    // @TODOMATT define/use a multiresolution constructor for Grid g (talk to max about this)
    Grid                  g(*backend, *dim, std::vector<std::function<bool(const Neon::index_3d&)>>{[](Neon::index_3d const& /*idx*/) { return true; }}, d3q19, Grid::Descriptor(depth));
    auto                  gridPtr = new (std::nothrow) Grid(g);
    AllocationCounter::Allocation();

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    handle = (uint64_t)gridPtr;
    std::cout << "grid_new - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}


auto mGrid_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "mGrid_delete - BEGIN" << std::endl;
    std::cout << "mGrid_delete - gridHandle " << handle << std::endl;

    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
        AllocationCounter::Deallocation();
    }
    handle = 0;

    std::cout << "mGrid_delete - END" << std::endl;
    return 0;
}

auto mGrid_get_span(
    uint64_t&                   gridHandle,
    uint64_t&                   grid_level,
    Neon::domain::mGrid::Span*  spanRes,
    int                         execution,
    int                         device,
    int                         data_view)
    -> int
{
    std::cout << "mGrid_get_span - BEGIN " << std::endl;
    std::cout << "mGrid_get_span - gridHandle " << gridHandle << std::endl;
    std::cout << "mGrid_get_span - grid_level " << grid_level << std::endl;
    std::cout << "mGrid_get_span - execution " << execution << std::endl;
    std::cout << "mGrid_get_span - device " << device << std::endl;
    std::cout << "mGrid_get_span - data_view " << data_view << std::endl;
    std::cout << "mGrid_get_span - Span size " << sizeof(*spanRes) << std::endl;

    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        if (grid_level < grid.getGridCount()) {
            std::cout << "grid_level out of range in mGrid_get_span" << std::endl;
        }
        auto& gridSpan = grid(grid_level).getSpan(Neon::ExecutionUtils::fromInt(execution),
                                      device,
                                      Neon::DataViewUtil::fromInt(data_view));
        (*spanRes) = gridSpan;
        std::cout << "mGrid_get_span - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

auto mGrid_mField_new(
    uint64_t& handle,
    uint64_t& gridHandle)
    -> int
{
    std::cout << "mGrid_mField_new - BEGIN" << std::endl;
    std::cout << "mGrid_mField_new - gridHandle " << gridHandle << std::endl;
    std::cout << "mGrid_mField_new - handle " << handle << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 0>;
    Grid* gridPtr = reinterpret_cast<Grid*>(handle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        Field field = grid.newField<int, 0>("test", 1, 0, Neon::DataUse::HOST_DEVICE);
        Field* fieldPtr = new (std::nothrow) Field(field);
        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }
        AllocationCounter::Allocation();
        handle = (uint64_t)fieldPtr;
        std::cout << "mGrid_mField_new - END " << handle << std::endl;

        return 0;
    }
    std::cout << "mGrid_mField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

auto mGrid_mField_get_partition(
    uint64_t&                                                   field_handle,
    [[maybe_unused]] Neon::domain::mGrid::Partition<int, 0>*    partitionPtr,
    uint64_t&                                                   field_level,
    Neon::Execution                                             execution,
    int                                                         device,
    Neon::DataView                                              data_view)
    -> int
{

    std::cout << "mGrid_mField_get_partition - BEGIN " << std::endl;
    std::cout << "mGrid_mField_get_partition - field_handle " << field_handle << std::endl;
    std::cout << "mGrid_mField_get_partition - execution " << Neon::ExecutionUtils::toString(execution) << std::endl;
    std::cout << "mGrid_mField_get_partition - field_level " << field_level << std::endl;
    std::cout << "mGrid_mField_get_partition - device " << device << std::endl;
    std::cout << "mGrid_mField_get_partition - data_view " << Neon::DataViewUtil::toString(data_view) << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 0>;

    Field* fieldPtr = (Field*)field_handle;

    if (fieldPtr != nullptr) {
        const auto& descriptor = fieldPtr->getDescriptor();

        // check to make sure that the given field level is within bounds. The first clause is to allow a cast in the second clause.
        if (descriptor.getDepth() < 0 || field_level >= static_cast<uint64_t>(descriptor.getDepth())) {
            std::cout << "field index out of bounds" << std::endl;
            return -1;
        }
        auto p = (*fieldPtr)(field_level).getPartition(execution,
                                        device,
                                        data_view);
        std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;

        std::cout << "mGrid_mField_get_partition - END" << std::endl;

        return 0;
    }
    return -1;
}

auto mGrid_mField_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "mGrid_mField_delete - BEGIN" << std::endl;
    std::cout << "mGrid_mField_delete - handle " << handle << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = (Field*)handle;

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Allocation();
    }
    handle = 0;
    std::cout << "mGrid_mField_delete - END" << std::endl;

    return 0;
}

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

auto mGrid_get_properties( /* TODOMATT verify what the return of this method should be */
    uint64_t& gridHandle,
    uint64_t& grid_level,
    const Neon::index_3d* idx) 
    -> int
{
    std::cout << "mGrid_get_properties begin" << std::endl;
    
    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    if (grid_level >= gridPtr->getGridCount()) {
            std::cout << "grid_level out of range in mGrid_get_properties" << std::endl;
        }
    int returnValue = int((*gridPtr)(grid_level).getProperties(*idx).getDataView());
    std::cout << "mGrid_get_properties end" << std::endl;

    return returnValue;
}

auto mGrid_is_inside_domain(
    uint64_t& gridHandle,
    uint64_t& grid_level,
    const Neon::index_3d* idx
    ) 
    -> bool
{
    std::cout << "mGrid_is_inside_domain begin" << std::endl;
    
    using Grid = Neon::domain::mGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    bool returnValue = gridPtr->isInsideDomain(*idx, grid_level);

    std::cout << "mGrid_is_inside_domain end" << std::endl;


    return returnValue;
}

auto mGrid_mField_read(
    uint64_t& fieldHandle,
    uint64_t& field_level,
    const Neon::index_3d* idx,
    const int& cardinality)
    -> int
{
    std::cout << "mGrid_mField_read begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(*idx, cardinality, field_level);
    
    std::cout << "mGrid_mField_read end" << std::endl;

    return returnValue;
}

auto mGrid_mField_write(
    uint64_t& fieldHandle,
    uint64_t& field_level,
    const Neon::index_3d* idx,
    const int& cardinality,
    int newValue)
    -> int
{
    std::cout << "mGrid_mField_write begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->getReference(*idx, cardinality, field_level) = newValue;
    
    std::cout << "mGrid_mField_write end" << std::endl;
    return 0;
}

auto mGrid_mField_update_host_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int
{
    std::cout << "mGrid_mField_update_host_data begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateHostData(streamSetId);
    
    std::cout << "mGrid_mField_update_host_data end" << std::endl;
    return 0;
}

auto mGrid_mField_update_device_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int
{
    std::cout << "mGrid_mField_update_device_data begin" << std::endl;

    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateDeviceData(streamSetId);
    
    std::cout << "mGrid_mField_update_device_data end" << std::endl;
    return 0;
}