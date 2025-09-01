#include "Neon/py/dGrid.h"
#include "Neon/py/macros.h"

#include <nvtx3/nvToolsExt.h>
#include "Neon/domain/Grids.h"
#include "Neon/py/AllocationCounter.h"
#include "Neon/py/macros.h"
#include "Neon/set/Backend.h"

#include "Neon/domain/operator.h"

// Workaround for CUDA 12.8 namespace issue
#if defined(__CUDACC__) && CUDA_VERSION >= 12080 && CUDA_VERSION < 13000
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#include <cuda/std/functional>
#include <cuda/std/utility>
#endif
template <typename T>
auto mGrid_mField_fill(
    void* fieldHandle,
    int level,
    T     value,
    int   streamSetId)
    -> int
{
    NEON_PY_PRINT_BEGIN(fieldHandle);


    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field* fieldPtr = reinterpret_cast<Field*>(fieldHandle);

    if (fieldPtr == nullptr) {
        std::cout << "invalid field" << std::endl;
        return -1;
    }
    const bool isMultires= true;
    auto container = Neon::domain::newfillContainer<Field, isMultires>(value, *fieldPtr, level);
    container.run(streamSetId);
    return 0;
}

DO_EXPORT(int8, 4, mGrid_mField_fill, int, void*, fieldHandle, int, level,int8, value, int, streamIdx);
DO_EXPORT(uint8, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, uint8, value, int, streamIdx);
DO_EXPORT(bool, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, bool, value, int, streamIdx);

DO_EXPORT(int32, 4, mGrid_mField_fill, int, void*, fieldHandle, int, level,int32, value, int, streamIdx);
DO_EXPORT(uint32, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, uint32, value, int, streamIdx);

DO_EXPORT(int64, 4, mGrid_mField_fill, int, void*, fieldHandle, int, level,int64, value, int, streamIdx);
DO_EXPORT(uint64, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, uint64, value, int, streamIdx);

DO_EXPORT(float32, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, float32, value, int, streamIdx);
DO_EXPORT(float64, 4, mGrid_mField_fill, int, void*, fieldHandle,int, level, float64, value, int, streamIdx);

template <typename T>
auto mGrid_mField_copy(
    void* fieldHandleDst,
    void* fieldHandleSrc,
    int level, 
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
    using Grid = Neon::domain::mGrid;
    using Field = Grid::Field<T, 0>;

    Field&       fieldDst = *reinterpret_cast<Field*>(fieldHandleDst);
    Field const& fieldSrc = *reinterpret_cast<Field*>(fieldHandleSrc);

    const bool isMultires= true;
    auto container = Neon::domain::newCopyContainer<Field, isMultires>(fieldSrc, fieldDst, level);
    container.run(streamSetId);
    return 0;
}

DO_EXPORT(int8, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);
DO_EXPORT(uint8, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);
DO_EXPORT(bool, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);

DO_EXPORT(int32, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB,int, level, int, streamIdx);
DO_EXPORT(uint32, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB,int, level, int, streamIdx);

DO_EXPORT(int64, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);
DO_EXPORT(uint64, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB,int, level, int, streamIdx);

DO_EXPORT(float32, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);
DO_EXPORT(float64, 4, mGrid_mField_copy, int, void*, fhA, void*, fhB, int, level,int, streamIdx);
