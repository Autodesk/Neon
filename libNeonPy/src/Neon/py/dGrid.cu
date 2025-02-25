#include "Neon/py/dGrid.h"
#include "Neon/py/macros.h"

#include <nvtx3/nvToolsExt.h>
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"
#include "Neon/set/Backend.h"

#include "Neon/domain/operator.h"
template <typename T>
auto dGrid_dField_fill(
    void* fieldHandle,
    T     value,
    int   streamSetId)
    -> int
{
    NEON_PY_PRINT_BEGIN(fieldHandle);

    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    auto container = Neon::domain::newfillContainer(value, *fieldPtr);
    container.run(streamSetId);
    return 0;
}

DO_EXPORT(int8, 3, dGrid_dField_fill, int, void*, fieldHandle, int8, value, int, streamIdx);
DO_EXPORT(uint8, 3, dGrid_dField_fill, int, void*, fieldHandle, uint8, value, int, streamIdx);
DO_EXPORT(bool, 3, dGrid_dField_fill, int, void*, fieldHandle, bool, value, int, streamIdx);

DO_EXPORT(int32, 3, dGrid_dField_fill, int, void*, fieldHandle, int32, value, int, streamIdx);
DO_EXPORT(uint32, 3, dGrid_dField_fill, int, void*, fieldHandle, uint32, value, int, streamIdx);

DO_EXPORT(int64, 3, dGrid_dField_fill, int, void*, fieldHandle, int64, value, int, streamIdx);
DO_EXPORT(uint64, 3, dGrid_dField_fill, int, void*, fieldHandle, uint64, value, int, streamIdx);

DO_EXPORT(float32, 3, dGrid_dField_fill, int, void*, fieldHandle, float32, value, int, streamIdx);
DO_EXPORT(float64, 3, dGrid_dField_fill, int, void*, fieldHandle, float64, value, int, streamIdx);

template <typename T>
auto dGrid_dField_copy(
    void* fieldHandleDst,
    void* fieldHandleSrc,
    int   streamSetId)
    -> int
{
    NEON_PY_PRINT_BEGIN(fieldHandleDst);
    if (fieldHandleDst == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    if (fieldHandleSrc == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    using Grid = Neon::dGrid;
    using Field = Grid::Field<T, 0>;

    Field&       fieldDst = *reinterpret_cast<Field*>(fieldHandleDst);
    Field const& fieldSrc = *reinterpret_cast<Field*>(fieldHandleSrc);

    auto container = Neon::domain::newCopyContainer(fieldSrc, fieldDst);
    container.run(streamSetId);
    return 0;
}

DO_EXPORT(int8, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
DO_EXPORT(uint8, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
DO_EXPORT(bool, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);

DO_EXPORT(int32, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
DO_EXPORT(uint32, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);

DO_EXPORT(int64, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
DO_EXPORT(uint64, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);

DO_EXPORT(float32, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
DO_EXPORT(float64, 3, dGrid_dField_copy, int, void*, fhA, void*, fhB, int, streamIdx);
