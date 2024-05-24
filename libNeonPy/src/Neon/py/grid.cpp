#include "Neon/py/grid.h"
#include "Neon/domain/Grids.h"

auto dGrid_new(
    uint64_t& handle)
    -> int
{
    std::cout << "dGrid_new - BEGIN" << std::endl;
    std::cout << "dGrid_get_span - gridHandle " << handle << std::endl;

    Neon::init();

    using Grid = Neon::dGrid;
    Neon::Backend         bk(1, Neon::Runtime::openmp);
    Neon::index_3d        dim(10, 10, 10);
    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);
    Grid                  g(bk, dim, [](Neon::index_3d const& /*idx*/) { return true; }, d3q19);
    auto                  gridPtr = new (std::nothrow) Grid(g);

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    handle = (uint64_t)gridPtr;
    std::cout << "grid_new - END" << std::endl;

    // g.ioDomainToVtk("")
    return 0;
}

auto dGrid_delete(
    uint64_t& handle)
    -> int
{
    std::cout << "dGrid_delete - BEGIN" << std::endl;
    std::cout << "dGrid_get_span - gridHandle " << handle << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)handle;

    if (gridPtr != nullptr) {
        delete gridPtr;
    }
    handle = 0;
    std::cout << "dGrid_delete - END" << std::endl;
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
        std::cout << "field_new - END" << &gridSpan << std::endl;

        return 0;
    }
    return -1;
}

auto dGrid_dField_new(
    uint64_t& handle,
    uint64_t& gridHandle)
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
        Field field = grid.newField<int, 0>("test", 1, 0, Neon::DataUse::HOST_DEVICE);
        std::cout << field.toString() << std::endl;
        Field* fieldPtr = new (std::nothrow) Field(field);
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
    }
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