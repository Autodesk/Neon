#pragma once
#include "Neon/core/core.h"

namespace Neon::domain {
struct Stencil
{

   public:
    /**
     * Constructing a stencil by providing a list of points.
     * A flag can be enable to filter out a possible <0,0,0> point.
     *
     * @param listOfPoinst
     * @param filterCenterOut
     */
    Stencil(std::vector<Neon::index_3d> const& listOfPoinst,
            bool                               filterCenterOut = true);

    /**
     * Default constructor
     */
    Stencil() = default;

    /**
     * Returns number of point in the stencil
     * @return
     */
    auto nPoints()
        const -> int;

    /*
     * Returns the index of the direction in the vector of points.
     * -1 is returned if the 3dIndex is not found.
     */
    auto find(const Neon::index_3d& direction)
        const -> int;

    /**
     * Returns number of neighbours for the stencil
     * @return
     */
    auto nNeighbours()
        const -> int;

    auto points()
        const -> const std::vector<Neon::index_3d>&;

    auto neighbours()
        const -> const std::vector<Neon::index_3d>&;

    auto addPoint(const Neon::index_3d& newPoint)
        -> void;

    auto getRadius() const -> int32_t;

    /**
     * static method to create a 19 point stencil
     *
     * @param filterCenterOut
     * @return
     */
    static auto s19_t(bool filterCenterOut = true)
        -> Stencil;

    /**
     * static method to create a Laplace 7 point stencil
     *
     * @param filterCenterOut
     * @return
     */
    static auto s7_Laplace_t(bool filterCenterOut = true)
        -> Stencil;

    /**
     * static method to create a 27 point stencil
     *
     * @param filterCenterOut
     * @return
     */
    static auto s27_t(bool filterCenterOut = true)
        -> Stencil;

    /**
     * static method to create a Laplace 7 point stencil
     *
     * @param filterCenterOut
     * @return
     */
    static auto s6_Jacobi_t()
        -> Stencil;

    static auto getUnion(const std::vector<Stencil>& vec)
        -> Stencil;

   private:
    std::vector<Neon::index_3d> mPoints{};                                         /** point in the stencil */
    std::vector<Neon::index_3d> mNeighbours{};                                      /** point in the stencil excluding center position */
    void                        p_updateNeighbourList(bool filterCenterOut = true); /** helper fun to update m_neighbour base on m_points */
};
}  // namespace Neon::domain
