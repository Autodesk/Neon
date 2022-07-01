#pragma once

namespace Neon {
namespace solver {

/**
 * The MatVec class represents a matrix-vector multiply operation
 * It is expected to perform the necessary halo updates along with the MatVec operation
 *
 * @tparam Grid Type of the grid where this operation will be executed
 * @tparam Real Real value type (double or float)
 */
template <typename Grid_, typename Real>
class MatVec
{
   public:
    using self_t = MatVec<Grid_, Real>;
    using Grid = Grid_;
    using Field = typename Grid::template Field<Real>;
    using bdField = typename Grid::template Field<int8_t>;

    /**
     * Default constructor
     */
    MatVec() = default;

    /**
     * Virtual destructor to derived classes
     */
    virtual ~MatVec() = default;

    /**
     * Implementation of the matrix-vector operation A * x
     * @param[in] backend Backend specifying the device type and device set to use
     * @param[in] input Real valued input field x
     * @param[in] bd int8_t valued field marking Dirichlet boundary with 1 and 0 otherwise
     * @param[inout] output Real valued field holding the output A * x
     */
    virtual Neon::set::Container matVec(const Field& input, const bdField& bd, Field& output) = 0;
};

}  // namespace solver
}  // namespace Neon