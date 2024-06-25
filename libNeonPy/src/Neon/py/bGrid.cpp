#include "Neon/domain/Grids.h"
#include "Neon/py/bGrid.h"
#include "Neon/py/AllocationCounter.h"


auto bGrid_new(
    uint64_t& handle,
    uint64_t& backendPtr,
    Neon::index_3d dim)
    -> int
{
    std::cout << "bGrid_new - BEGIN" << std::endl;
    std::cout << "bGrid_new - gridHandle " << handle << std::endl;

    Neon::init();

    using Grid = Neon::bGrid;
    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    Grid                  g(*reinterpret_cast<Neon::Backend*>(backendPtr), dim, [](Neon::index_3d const& /*idx*/) { return true; }, d3q19);
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

auto bGrid_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "bGrid_delete - BEGIN" << std::endl;
    std::cout << "bGrid_delete - gridHandle " << handle << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
        AllocationCounter::Deallocation();
    }
    handle = 0;
    std::cout << "bGrid_delete - END" << std::endl;
    return 0;
}

auto bGrid_get_span(
    uint64_t&          gridHandle,
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
        std::cout << "field_new - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

auto bGrid_bField_new(
    uint64_t& handle,
    uint64_t& gridHandle)
    -> int
{
    std::cout << "bGrid_bField_new - BEGIN" << std::endl;
    std::cout << "bGrid_bField_new - gridHandle " << gridHandle << std::endl;
    std::cout << "bGrid_bField_new - handle " << handle << std::endl;

    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    Grid& grid = *gridPtr;

    if (gridPtr != nullptr) {
        using Field = Grid::Field<int, 0>;
        Field field = grid.newField<int, 0>("test", 1, 0, Neon::DataUse::HOST_DEVICE);
        std::cout << field.toString() << std::endl;
        Field* fieldPtr = new (std::nothrow) Field(field);
        AllocationCounter::Allocation();

        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
            return -1;
        }
        handle = (uint64_t)fieldPtr;
        std::cout << "bGrid_bField_new - END " << handle << std::endl;

        return 0;
    }
    std::cout << "bGrid_bField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

auto bGrid_bField_get_partition(
    uint64_t&                                        field_handle,
    [[maybe_unused]] Neon::bGrid::Partition<int, 0>* partitionPtr,
    Neon::Execution                                  execution,
    int                                              device,
    Neon::DataView                                   data_view)
    -> int
{

    std::cout << "bGrid_bField_get_partition - BEGIN " << std::endl;
    std::cout << "bGrid_bField_get_partition - field_handle " << field_handle << std::endl;
    std::cout << "bGrid_bField_get_partition - execution " << Neon::ExecutionUtils::toString(execution) << std::endl;
    std::cout << "bGrid_bField_get_partition - device " << device << std::endl;
    std::cout << "bGrid_bField_get_partition - data_view " << Neon::DataViewUtil::toString(data_view) << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 0>;

    Field* fieldPtr = (Field*)field_handle;

    if (fieldPtr != nullptr) {
        auto p = fieldPtr->getPartition(execution,
                                        device,
                                        data_view);
        std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;

        std::cout << "bGrid_bField_get_partition - END" << std::endl;

        return 0;
    }
    return -1;
}

auto bGrid_bField_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "bGrid_bField_delete - BEGIN" << std::endl;
    std::cout << "bGrid_bField_delete - handle " << handle << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = (Field*)handle;

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Deallocation();
    }
    std::cout << "bGrid_bField_delete - END" << std::endl;

    return 0;
}

auto bGrid_span_size(
    Neon::bGrid::Span* spanRes)
    -> int
{
    return sizeof(*spanRes);
}

auto bGrid_bField_partition_size(
    Neon::bGrid::Partition<int, 0>* partitionPtr)
    -> int
{
    return sizeof(*partitionPtr);
}

auto bGrid_get_properties( /* TODOMATT verify what the return of this method should be */
    uint64_t& gridHandle,
    const Neon::index_3d& idx) 
    -> int
{
    std::cout << "bGrid_get_properties begin" << std::endl;
    
    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    int returnValue = int(gridPtr->getProperties(idx).getDataView());
    std::cout << "bGrid_get_properties end" << std::endl;

    return returnValue;
}

auto bGrid_is_inside_domain(
    uint64_t& gridHandle,
    const Neon::index_3d& idx) 
    -> bool
{
    std::cout << "bGrid_is_inside_domain begin" << std::endl;
    
    using Grid = Neon::bGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);
    bool returnValue = gridPtr->isInsideDomain(idx);

    std::cout << "bGrid_is_inside_domain end" << std::endl;


    return returnValue;
}

auto bGrid_bField_read(
    uint64_t& fieldHandle,
    const Neon::index_3d& idx,
    const int& cardinality)
    -> int
{
    std::cout << "bGrid_bField_read begin" << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(idx, cardinality);
    
    std::cout << "bGrid_bField_read end" << std::endl;

    return returnValue;
}

auto bGrid_bField_write(
    uint64_t& fieldHandle,
    const Neon::index_3d& idx,
    const int& cardinality,
    int newValue)
    -> int
{
    std::cout << "bGrid_bField_write begin" << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->getReference(idx, cardinality) = newValue;
    
    std::cout << "bGrid_bField_write end" << std::endl;
    return 0;
}

auto bGrid_bField_update_host_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int
{
    std::cout << "bGrid_bField_update_host_data begin" << std::endl;

    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateHostData(streamSetId);
    
    std::cout << "bGrid_bField_update_host_data end" << std::endl;
    return 0;
}

auto bGrid_bField_update_device_data(
    uint64_t& fieldHandle,
    int streamSetId)
    -> int
{
    std::cout << "bGrid_bField_update_device_data begin" << std::endl;
    
    using Grid = Neon::bGrid;
    using Field = Grid::Field<int, 1>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->updateDeviceData(streamSetId);
    
    std::cout << "bGrid_bField_update_device_data end" << std::endl;
    return 0;
}

extern "C" auto bGrid_bSpan_get_member_field_offsets(std::size_t* length)
    -> std::size_t*
{
    std::vector<std::size_t> offsets = Neon::domain::details::bGrid::bSpan<Neon::domain::details::bGrid::BlockDefault>::getOffsets(); // @TODOMATT I am not sure if it should be templated with <int> or with something else, but I think so?
    *length = offsets.size();
    return offsets.data();
}