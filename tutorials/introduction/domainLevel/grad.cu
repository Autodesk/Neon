#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "expandSphere.h"

/**
 * Function that generates a Neon Container to compute the grad over a scalar field.
 * Note: in Neon only constant field can be used as input for stencil computation
 */
template <typename Field>
auto computeGrad(const Field& levelSetField /** input scalar field we want to compute the grad.*/,
                 Field&       gradField /** input scalar field we want to compute the grad.*/,
                 double       h)
    -> Neon::set::Container
{
    if (levelSetField.getCardinality() != 1 || gradField.getCardinality() != 3) {
        NEON_THROW_UNSUPPORTED_OPERATION("Wrong cardinality detected.");
    }

    // The following Neon compute-lambda works with the assumption that the first elements of the stencil
    // given to the grid initialization are as follow:
    //
    //      {1, 0, 0},
    //      {0, 1, 0},
    //      {0, 0, 1},
    //      {-1, 0, 0},
    //      {0, -1, 0},
    //      {0, 0, -1}
    return levelSetField.getGrid().getContainer(
        "computeGrad", [&, h](Neon::set::Loader& L) {
            // Loading the sdf field for a stencil type of computation
            // as we will be using a 6 point stencil to compute the gradient
            auto&      levelSet = L.load(levelSetField, Neon::Compute::STENCIL);
            auto&      grad = L.load(gradField);

            // We can nicely compute the inverse of the spacing in the loading lambda
            const auto twiceOverH = 1. / h;


            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Cell& cell) mutable {
                // Central difference
                for (int i = 0; i < 3; i++) {
                    auto upIdx = i;
                    auto dwIdx = i + 3;

                    auto [valUp, isValidUp] = levelSet.nghVal(cell, upIdx, 0, 0);
                    auto [valDw, isValidDw] = levelSet.nghVal(cell, dwIdx, 0, 0);

                    if (!isValidUp || !isValidDw) {
                        grad(cell, 0) = 0;
                        grad(cell, 1) = 0;
                        grad(cell, 2) = 0;
                        break;
                    } else {
                        grad(cell, i) = (valUp - valDw) / twiceOverH;
                    }
                }
            };
        });
}

template auto computeGrad<Neon::domain::eGrid::Field<double, 0>>(const Neon::domain::eGrid::Field<double, 0>& levelSet, Neon::domain::eGrid::Field<double, 0>& grad, double h) -> Neon::set::Container;