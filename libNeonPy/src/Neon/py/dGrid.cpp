#include "Neon/py/dGrid.h"
#include "Neon/py/macros.h"

#include <nvtx3/nvToolsExt.h>
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/set/Backend.h"

auto dGrid_new(
    void**                handle,
    void*                 backendPtr,
    const Neon::index_3d* dim,
    int const*            sparsity_pattern,
    int                   numStencilPoints,
    int const*            stencilPointFlatArray)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);


    Neon::init();

    using Grid = Neon::dGrid;

    Neon::Backend* backend = reinterpret_cast<Neon::Backend*>(backendPtr);
    if (backend == nullptr) {
        std::cerr << "Invalid backend pointer" << std::endl;
        return -1;
    }

    Neon::domain::Stencil d3q19 = Neon::domain::Stencil::s19_t(false);

    std::vector<Neon::index_3d> points(numStencilPoints);
    for (int sId = 0; sId < numStencilPoints; sId++) {
        points[sId].x = stencilPointFlatArray[sId * 3];
        points[sId].y = stencilPointFlatArray[sId * 3 + 1];
        points[sId].z = stencilPointFlatArray[sId * 3 + 2];
    }

    Neon::domain::Stencil stencil(points);
    Grid                  g(
        *backend,
        *dim,
        [=](Neon::index_3d const& idx) {
            return sparsity_pattern[idx.x * (dim->x * dim->y) + idx.y * dim->z + idx.z];
        },
        stencil);
    auto gridPtr = new (std::nothrow) Grid(g);
    AllocationCounter::Allocation();

    if (gridPtr == nullptr) {
        std::cout << "NeonPy: Initialization error. Unable to allocage grid " << std::endl;
        return -1;
    }
    *handle = reinterpret_cast<void*>(gridPtr);
    NEON_PY_PRINT_END(*handle);

    // g.ioDomainToVtk("")
    return 0;
}

auto dGrid_delete(
    void** handle)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);


    using Grid = Neon::dGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(*handle);

    if (gridPtr != nullptr) {
        delete gridPtr;
        AllocationCounter::Deallocation();
    }
    *handle = nullptr;
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto dGrid_get_dimensions(
    void*           gridHandle,
    Neon::index_3d* dim)
    -> int
{
    NEON_PY_PRINT_BEGIN(gridHandle);

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

    // g.ioDomainToVtk("")
    NEON_PY_PRINT_END(gridHandle);

    return 0;
}

auto dGrid_get_span(
    void*              gridHandle,
    Neon::dGrid::Span* spanRes,
    int                execution,
    int                device,
    int                data_view)
    -> int
{
    NEON_PY_PRINT_BEGIN(gridHandle);

    //    std::cout << "dGrid_get_span - BEGIN " << std::endl;
    //    std::cout << "dGrid_get_span - gridHandle " << gridHandle << std::endl;
    //    std::cout << "dGrid_get_span - execution " << execution << std::endl;
    //    std::cout << "dGrid_get_span - device " << device << std::endl;
    //    std::cout << "dGrid_get_span - data_view " << data_view << std::endl;
    //    std::cout << "dGrid_get_span - Span size " << sizeof(*spanRes) << std::endl;

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)gridHandle;

    if (gridPtr != nullptr) {
        Grid& grid = *gridPtr;
        auto& gridSpan = grid.getSpan(Neon::ExecutionUtils::fromInt(execution),
                                      device,
                                      Neon::DataViewUtil::fromInt(data_view));
        (*spanRes) = gridSpan;
        // std::cout << "dGrid_get_span - END" << &gridSpan << std::endl;

        return 0;
        NEON_PY_PRINT_END(gridHandle);
    }
    return -1;
    NEON_PY_PRINT_END(gridHandle);
}
#define NEON_PY_GET_EVEN_PARAMS_1(a)
#define NEON_PY_GET_EVEN_PARAMS_2(a, b, ...) b, NEON_PY_GET_EVEN_PARAMS_1(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_3(a, b, c, ...) b, NEON_PY_GET_EVEN_PARAMS_2(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_4(a, b, c, d, ...) b, NEON_PY_GET_EVEN_PARAMS_3(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_5(a, b, c, d, e, ...) b, NEON_PY_GET_EVEN_PARAMS_4(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_6(a, b, c, d, e, f, ...) b, NEON_PY_GET_EVEN_PARAMS_5(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_7(a, b, c, d, e, f, g, ...) b, NEON_PY_GET_EVEN_PARAMS_6(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_8(a, b, c, d, e, f, g, h, ...) b, NEON_PY_GET_EVEN_PARAMS_7(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_9(a, b, c, d, e, f, g, h, i, ...) b, NEON_PY_GET_EVEN_PARAMS_8(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_10(a, b, c, d, e, f, g, h, i, j, ...) b, NEON_PY_GET_EVEN_PARAMS_9(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_11(a, b, c, d, e, f, g, h, i, j, k, ...) b, NEON_PY_GET_EVEN_PARAMS_10(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_12(a, b, c, d, e, f, g, h, i, j, k, l, ...) b, NEON_PY_GET_EVEN_PARAMS_11(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_13(a, b, c, d, e, f, g, h, i, j, k, l, m, ...) b, NEON_PY_GET_EVEN_PARAMS_12(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_14(a, b, c, d, e, f, g, h, i, j, k, l, m, n, ...) b, NEON_PY_GET_EVEN_PARAMS_13(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, ...) b, NEON_PY_GET_EVEN_PARAMS_14(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...) b, NEON_PY_GET_EVEN_PARAMS_15(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_17(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, ...) b, NEON_PY_GET_EVEN_PARAMS_16(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_18(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, ...) b, NEON_PY_GET_EVEN_PARAMS_17(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_19(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, ...) b, NEON_PY_GET_EVEN_PARAMS_18(__VA_ARGS__)
#define NEON_PY_GET_EVEN_PARAMS_20(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, ...) b, NEON_PY_GET_EVEN_PARAMS_19(__VA_ARGS__)

// Main macro to select the correct GET_EVEN_PARAMS_X macro based on the number of arguments
#define NEON_PY_GET_EVEN_PARAMS(N, ...) NEON_PY_GET_EVEN_PARAMS##N(__VA_ARGS__)


// Base case: No more pairs to process
#define EXTRACT_NAMES_0()

// Recursive macros to extract variable names
#define EXTRACT_NAMES_1(TYPE1, NAME1) NAME1
#define EXTRACT_NAMES_2(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_1(__VA_ARGS__)
#define EXTRACT_NAMES_3(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_2(__VA_ARGS__)
#define EXTRACT_NAMES_4(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_3(__VA_ARGS__)
#define EXTRACT_NAMES_5(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_4(__VA_ARGS__)
#define EXTRACT_NAMES_6(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_5(__VA_ARGS__)
#define EXTRACT_NAMES_7(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_6(__VA_ARGS__)
#define EXTRACT_NAMES_8(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_7(__VA_ARGS__)
#define EXTRACT_NAMES_9(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_8(__VA_ARGS__)
#define EXTRACT_NAMES_10(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_9(__VA_ARGS__)
#define EXTRACT_NAMES_11(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_10(__VA_ARGS__)
#define EXTRACT_NAMES_12(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_11(__VA_ARGS__)
#define EXTRACT_NAMES_13(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_12(__VA_ARGS__)
#define EXTRACT_NAMES_14(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_13(__VA_ARGS__)
#define EXTRACT_NAMES_15(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_14(__VA_ARGS__)
#define EXTRACT_NAMES_16(TYPE1, NAME1, ...) NAME1, EXTRACT_NAMES_15(__VA_ARGS__)

// Main macro to select the correct EXTRACT_NAMES_X macro based on the number of pairs
#define EXTRACT_NAMES(N, ...) EXTRACT_NAMES_##N(__VA_ARGS__)


// Helper macro to create a single parameter from a TYPE and VARIABLE_NAME pair
#define PAIR(TYPE, NAME) TYPE NAME

// Recursive macros to handle multiple pairs, ensuring no trailing commas
#define EXPAND_PAIR_1(TYPE1, NAME1) PAIR(TYPE1, NAME1)
#define EXPAND_PAIR_2(TYPE1, NAME1, TYPE2, NAME2) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2)
#define EXPAND_PAIR_3(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3)
#define EXPAND_PAIR_4(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4)
#define EXPAND_PAIR_5(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5)
#define EXPAND_PAIR_6(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6)
#define EXPAND_PAIR_7(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7)
#define EXPAND_PAIR_8(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8)
#define EXPAND_PAIR_9(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9)
#define EXPAND_PAIR_10(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10)
#define EXPAND_PAIR_11(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11)
#define EXPAND_PAIR_12(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11, TYPE12, NAME12) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11), PAIR(TYPE12, NAME12)
#define EXPAND_PAIR_13(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11, TYPE12, NAME12, TYPE13, NAME13) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11), PAIR(TYPE12, NAME12), PAIR(TYPE13, NAME13)
#define EXPAND_PAIR_14(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11, TYPE12, NAME12, TYPE13, NAME13, TYPE14, NAME14) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11), PAIR(TYPE12, NAME12), PAIR(TYPE13, NAME13), PAIR(TYPE14, NAME14)
#define EXPAND_PAIR_15(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11, TYPE12, NAME12, TYPE13, NAME13, TYPE14, NAME14, TYPE15, NAME15) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11), PAIR(TYPE12, NAME12), PAIR(TYPE13, NAME13), PAIR(TYPE14, NAME14), PAIR(TYPE15, NAME15)
#define EXPAND_PAIR_16(TYPE1, NAME1, TYPE2, NAME2, TYPE3, NAME3, TYPE4, NAME4, TYPE5, NAME5, TYPE6, NAME6, TYPE7, NAME7, TYPE8, NAME8, TYPE9, NAME9, TYPE10, NAME10, TYPE11, NAME11, TYPE12, NAME12, TYPE13, NAME13, TYPE14, NAME14, TYPE15, NAME15, TYPE16, NAME16) PAIR(TYPE1, NAME1), PAIR(TYPE2, NAME2), PAIR(TYPE3, NAME3), PAIR(TYPE4, NAME4), PAIR(TYPE5, NAME5), PAIR(TYPE6, NAME6), PAIR(TYPE7, NAME7), PAIR(TYPE8, NAME8), PAIR(TYPE9, NAME9), PAIR(TYPE10, NAME10), PAIR(TYPE11, NAME11), PAIR(TYPE12, NAME12), PAIR(TYPE13, NAME13), PAIR(TYPE14, NAME14), PAIR(TYPE15, NAME15), PAIR(TYPE16, NAME16)

// Macro to select the correct EXPAND_PAIR_X macro based on the number of pairs (N)
#define EXPAND_PAIRS(N, ...) EXPAND_PAIR_##N(__VA_ARGS__)

// Helper macro to remove parentheses
#define UNPAREN(...) __VA_ARGS__

#define DO_EXPORT(TYPE, N, FOO_NAME, RET, ...)                             \
    extern "C" auto FOO_NAME##_##TYPE(EXPAND_PAIRS(N, __VA_ARGS__)) -> RET \
    {                                                                      \
        return FOO_NAME<TYPE>(EXTRACT_NAMES(N, __VA_ARGS__));              \
    }


template <typename T>
auto dGrid_dField_new(
    void** handle,
    void*  gridHandle,
    int    cardinality)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);

    using Grid = Neon::dGrid;
    Grid* gridPtr = (Grid*)gridHandle;

    if (gridPtr != nullptr) {
        Grid& grid = *gridPtr;

        using Field = Grid::Field<T, 0>;
        Field field = grid.newField<T, 0>("test", cardinality, 0, Neon::DataUse::HOST_DEVICE);
        std::cout << field.toString() << std::endl;

        Field* fieldPtr = new (std::nothrow) Field(field);
        AllocationCounter::Allocation();

        if (fieldPtr == nullptr) {
            std::cout << "NeonPy: Initialization error. Unable to allocate grid " << std::endl;
            return -1;
        }
        *handle = (void*)fieldPtr;
        NEON_PY_PRINT_END(*handle);

        return 0;
    }
    std::cout << "dGrid_dField_new - ERROR (grid ptr " << gridPtr << ") " << std::endl;

    return -1;
}

using int8 = int8_t;
using uint8 = uint8_t;

using int32 = int32_t;
using uint32 = uint32_t;

using int64 = int64_t;
using uint64 = uint64_t;

using float32 = float;
using float64 = double;


DO_EXPORT(int8, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint8, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(bool, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int32, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint32, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(int64, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(uint64, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);

DO_EXPORT(float32, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);
DO_EXPORT(float64, 3, dGrid_dField_new, int, void**, handle, void*, gridHandle, int, cardinality);

template <typename T>
auto dGrid_dField_delete(
    void** handle)
    -> int
{
    NEON_PY_PRINT_BEGIN(*handle);


    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = (Field*)(*handle);

    if (fieldPtr != nullptr) {
        delete fieldPtr;
        AllocationCounter::Deallocation();
    }

    *handle = nullptr;
    NEON_PY_PRINT_END(*handle);

    return 0;
}

DO_EXPORT(int8, 1, dGrid_dField_delete, int, void**, handle);
DO_EXPORT(uint8, 1, dGrid_dField_delete, int, void**, handle);
DO_EXPORT(bool, 1, dGrid_dField_delete, int, void**, handle);

DO_EXPORT(int32, 1, dGrid_dField_delete, int, void**, handle);
DO_EXPORT(uint32, 1, dGrid_dField_delete, int, void**, handle);

DO_EXPORT(int64, 1, dGrid_dField_delete, int, void**, handle);
DO_EXPORT(uint64, 1, dGrid_dField_delete, int, void**, handle);

DO_EXPORT(float32, 1, dGrid_dField_delete, int, void**, handle);
DO_EXPORT(float64, 1, dGrid_dField_delete, int, void**, handle);


template <typename T>
auto dGrid_dField_get_partition(
    void*                                          field_handle,
    [[maybe_unused]] Neon::dGrid::Partition<T, 0>* partitionPtr,
    Neon::Execution                                execution,
    int                                            device,
    Neon::DataView                                 data_view)
    -> int
{
    NEON_PY_PRINT_BEGIN(field_handle);


    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = (Field*)field_handle;
    // std::cout << fieldPtr->toString() << std::endl;

    if (fieldPtr != nullptr) {
        auto p = fieldPtr->getPartition(execution,
                                        device,
                                        data_view);
        // std::cout << p.cardinality() << std::endl;
        *partitionPtr = p;
        // std::cout << "dGrid_dField_get_partition\n"
        //     << partitionPtr->to_string();

        // std::cout << "dGrid_dField_get_partition - END" << std::endl;

        return 0;
        NEON_PY_PRINT_END(field_handle);
    }
    NEON_PY_PRINT_END(field_handle);

    return -1;
}

DO_EXPORT(int8, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<int8, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint8, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<uint8, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(bool, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<bool, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(int32, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<int32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint32, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<uint32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(int64, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<int64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(uint64, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<uint64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);

DO_EXPORT(float32, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<float32, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);
DO_EXPORT(float64, 5, dGrid_dField_get_partition, int, void*, field_handle, decltype(Neon::dGrid::Partition<float64, 0>())*, partitionPtr, Neon::Execution, execution, int, device, Neon::DataView, data_view);



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

auto dGrid_get_properties(/* TODOMATT verify what the return of this method should be */
                          void*                 gridHandle,
                          Neon::index_3d const* idx)
    -> int
{
    NEON_PY_PRINT_BEGIN(gridHandle);

    using Grid = Neon::dGrid;
    Grid* gridPtr = reinterpret_cast<Grid*>(gridHandle);

    int result = static_cast<int>(gridPtr->getProperties(*idx).getDataView());
    NEON_PY_PRINT_END(gridHandle);

    return result;
}

auto dGrid_is_inside_domain(
    void*                       gridHandle,
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

template <typename T>
auto dGrid_dField_read(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    const int             cardinality)
    -> T
{
    // std::cout << "dGrid_dField_read begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
    }

    auto returnValue = (*fieldPtr)(*idx, cardinality);

    // std::cout << "dGrid_dField_read end" << std::endl;

    return returnValue;
}

DO_EXPORT(int8, 3, dGrid_dField_read, int8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint8, 3, dGrid_dField_read, uint8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(bool, 3, dGrid_dField_read, uint8, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int32, 3, dGrid_dField_read, int32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint32, 3, dGrid_dField_read, uint32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(int64, 3, dGrid_dField_read, int64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(uint64, 3, dGrid_dField_read, uint64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

DO_EXPORT(float32, 3, dGrid_dField_read, float32, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);
DO_EXPORT(float64, 3, dGrid_dField_read, float64, void*, fieldHandle, const Neon::index_3d*, idx, const int, cardinality);

template <typename T>
auto dGrid_dField_write(
    void*                 fieldHandle,
    const Neon::index_3d* idx,
    int                   cardinality,
    T                     newValue)
    -> int
{
    // std::cout << "dGrid_dField_write begin" << std::endl;

    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }

    fieldPtr->getReference(*idx, cardinality) = newValue;

    // std::cout << "dGrid_dField_write end" << std::endl;
    return 0;
}

DO_EXPORT(int8, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int8, newValue);
DO_EXPORT(uint8, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint8, newValue);
DO_EXPORT(bool, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint8, newValue);

DO_EXPORT(int32, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int32, newValue);
DO_EXPORT(uint32, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint32, newValue);

DO_EXPORT(int64, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, int64, newValue);
DO_EXPORT(uint64, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, uint64, newValue);

DO_EXPORT(float32, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, float32, newValue);
DO_EXPORT(float64, 4, dGrid_dField_write, int, void*, fieldHandle, const Neon::index_3d*, idx, int, cardinality, float64, newValue);


template <typename T>
auto dGrid_dField_update_host_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
#ifdef NEON_USE_NVTX
    nvtxRangePush("dGrid_dField_update_host_data");
#endif

    NEON_PY_PRINT_BEGIN(fieldHandle);

    using Grid = Neon::dGrid;
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

DO_EXPORT(int8, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, dGrid_dField_update_host_data, int, void*, fieldHandle, int, streamSetId);


template <typename T>
auto dGrid_dField_update_device_data(
    void* fieldHandle,
    int   streamSetId)
    -> int
{
    NEON_PY_PRINT_BEGIN(fieldHandle);

#ifdef NEON_USE_NVTX
    nvtxRangePush("dGrid_dField_update_host_data");
#endif

    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = (Field*)fieldHandle;

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

DO_EXPORT(int8, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint8, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(bool, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int32, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint32, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(int64, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(uint64, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);

DO_EXPORT(float32, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);
DO_EXPORT(float64, 2, dGrid_dField_update_device_data, int, void*, fieldHandle, int, streamSetId);


extern "C" auto dGrid_dSpan_get_member_field_offsets(size_t* offsets,
                                                     size_t* length)
    -> void
{
    Neon::domain::details::dGrid::dSpan::getOffsets(offsets, length);
}

extern "C" auto dGrid_dField_dPartition_get_member_field_offsets(size_t* offsets, size_t* length)
    -> void
{
    Neon::domain::details::dGrid::dPartition<int, 0>::getOffsets(offsets, length);
}
