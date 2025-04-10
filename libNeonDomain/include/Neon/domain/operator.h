#pragma once

namespace Neon::domain {
template <typename Field, bool isMultires = false>
auto newfillContainer(typename Field::Type& val,
                      Field&                filedA,
                      [[maybe_unused]] int  level = -1)
    -> Neon::set::Container
{
    if constexpr (!isMultires) {
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
    } else {
        const auto& grid = filedA.getGrid();
        return grid.newContainer(
            "newfillContainer",
            level,
            [&, val](Neon::set::Loader& loader) {
                auto& a = filedA.load(loader, level, Neon::MultiResCompute::MAP);

                return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                    for (int i = 0; i < a.cardinality(); i++) {
                        a(e, i) = val;
                    }
                };
            });
    }
}


template <typename Field, bool isMultires = false>
auto newCopyContainer(const Field&         filedSrc,
                      Field&               fieldDst,
                      [[maybe_unused]] int level = -1)
    -> Neon::set::Container
{
    if constexpr (!isMultires) {
        const auto& grid = filedSrc.getGrid();
        return grid.newContainer(
            "newCopyContainer",
            [&](Neon::set::Loader& loader) {
                const auto a = loader.load(filedSrc);
                auto       b = loader.load(fieldDst);

                return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                    for (int i = 0; i < a.cardinality(); i++) {
                        // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                        b(e, i) = a(e, i);
                    }
                };
            });
    } else {
        auto& fieldSrc_level = filedSrc(level);
        auto& fieldDst_level = fieldDst(level);

        const auto& grid = filedSrc.getGrid();
        return grid.newContainer(
            "newCopyContainer",
            level,
            [&](Neon::set::Loader& loader) {
                const auto a = filedSrc.load(loader, level,  Neon::MultiResCompute::MAP);
                auto       b = fieldDst.load(loader, level, Neon::MultiResCompute::MAP);

                return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                    for (int i = 0; i < a.cardinality(); i++) {
                        // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                        b(e, i) = a(e, i);
                    }
                };
            });
    }
}

}  // namespace Neon::domain