
#include "gtest/gtest.h"
#include "Neon/skeleton/Skeleton.h"
#include "Neon/domain/eGrid.h"
TEST(skeleton, init)
{
    Neon::skeleton::Skeleton s;
    (void)s;
// Some ideas of what it will be the API
#if 0

    if constexpr (0){
        Neon::domain::details::eGrid::eGrid_t g;
        Neon::domain::details::eGrid::eGrid_t::field_t<double> A;
        Neon::domain::details::eGrid::eGrid_t::field_t<double> B;
        auto kenel = ...
        s.userCode([]{
            s.forEach(IN g, kernel);
        });
    }

    /// implementation of userCode
    void forEach(IN g, kernel){
    struct_t kernelParameterInfo {
        fieldId, USE_MODE (R or W), OP(Map, Stencil)
    };
    std::vector<>
    dependencyCapture_t depCaputure()
        kernel(depCaputure);
    }

#endif
}