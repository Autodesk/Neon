#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "expandSphere.h"

template <typename Field>
auto computeGrad(const Field& sdfField,
                      Field&       gradField,
                      double       h)
    -> Neon::set::Container
{
    if(sdfField.getCardinality() != 1 || gradField.getCardinality()!= 3 ){
        NEON_THROW_UNSUPPORTED_OPERATION("Wrong cardinality detected.");
    }

    return sdfField.getGrid().getContainer(
        "computeGrad", [&, h](Neon::set::Loader& L) {
            auto&      sdf = L.load(sdfField);
            auto&      grad = L.load(gradField, Neon::Compute::STENCIL);
            const auto twiceOverH = 1. / h;


            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Cell& cell) mutable {
                // Central difference
                for (int i = 0; i < 3; i++) {
                    auto upIdx = i;
                    auto dwIdx = i + 3;

                    auto [valUp, isValidUp] = sdf.nghVal(cell, upIdx, 0, 0);
                    auto [valDw, isValidDw] = sdf.nghVal(cell, dwIdx, 0, 0);

                    if (!isValidUp || !isValidDw) {
                        grad(cell, 0) = 0;
                        grad(cell, 1) = 0;
                        grad(cell, 2) = 0;
                        break ;
                    } else {
                        grad(cell, i) = (valUp - valDw) / twiceOverH;
                    }
                }
            };
        });
}

template auto computeGrad<Neon::domain::eGrid::Field<double, 0>>(const Neon::domain::eGrid::Field<double, 0>& sdf, Neon::domain::eGrid::Field<double, 0>& grad, double h) -> Neon::set::Container;