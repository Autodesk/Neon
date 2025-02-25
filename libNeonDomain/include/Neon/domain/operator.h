#pragma once

namespace Neon::domain {
template <typename Field>
auto newfillContainer(typename Field::Type& val,
                      Field&                filedA)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "newfillContainer",
        [&, val](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    a(e, i) = val;
                }
            };
        });
}


template <typename Field>
auto newCopyContainer(typename Field::Type& val,
                      const Field&          filedSrc,
                      Field&                fieldDst)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "newCopyContainer",
        [&, val](Neon::set::Loader& loader) {
            const auto a = loader.load(filedSrc);
            auto       b = loader.load(fieldDst);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    b(e, i) = a(e, i);
                }
            };
        });
}
}  // namespace Neon::domain