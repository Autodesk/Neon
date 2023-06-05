// References
// 2D LBM: https://github.com/hietwll/LBM_Taichi
// 2D LBM Verification data: https://www.sciencedirect.com/science/article/pii/0021999182900584
//For 2D/3D constants: https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods

#include <iomanip>
#include <sstream>

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/skeleton/Skeleton.h"

enum class FlowType
{
    border = 0,
    obstacle = 1,
};

template <unsigned int DIM, unsigned int COMP>
Neon::domain::Stencil create_stencil();

template <>
Neon::domain::Stencil create_stencil<2, 9>()
{
    std::vector<Neon::index_3d> stencil;
    stencil.reserve(9);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            stencil.emplace_back(Neon::index_3d(x, y, 0));
        }
    }
    return Neon::domain::Stencil(stencil);
}

template <>
Neon::domain::Stencil create_stencil<3, 19>()
{
    // filterCenterOut = false;
    return Neon::domain::Stencil::s19_t(false);
}

template <typename Field>
inline void exportVTI(const int t, Field& field)
{
    printf("\n Exporting Frame =%d", t);
    int                precision = 4;
    std::ostringstream oss;
    oss << std::setw(precision) << std::setfill('0') << t;
    std::string prefix = "lbm" + std::to_string(field.getCardinality()) + "D_";
    std::string fname = prefix + oss.str();
    field.ioToVtk(fname, "field");
}


/**
 * Get the x, y, or z component of the lattice vector 
 * represented by the component k.
 */
template <unsigned int DIM>
NEON_CUDA_HOST_DEVICE int get_e(const int k, const int id)
{
    static_assert(DIM == 2 || DIM == 3, "Dimension has to be either 2 or 3");

    if constexpr (DIM == 2) {
        static int array_x[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        static int array_y[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
        switch (id) {
            case 0:
                return array_x[k];
            case 1:
                return array_y[k];
            default:
                return 0;
        }
    }

    if constexpr (DIM == 3) {
        static int arr[19][3] = {
            {0, 0, 0},

            {1, 0, 0},
            {-1, 0, 0},
            {0, 1, 0},
            {0, -1, 0},
            {0, 0, 1},
            {0, 0, -1},

            {1, 1, 0},
            {-1, 1, 0},
            {1, -1, 0},
            {-1, -1, 0},
            {1, 0, 1},
            {-1, 0, 1},
            {1, 0, -1},
            {-1, 0, -1},
            {0, 1, 1},
            {0, -1, 1},
            {0, 1, -1},
            {0, -1, -1},
        };
        return arr[k][id];
    }
}


/**
 * Get the weight that corresponds to the 
 * lattice component represented by the component k.
 */
template <unsigned int DIM>
NEON_CUDA_HOST_DEVICE double get_w(const int k)
{
    static_assert(DIM == 2 || DIM == 3, "Dimension has to be either 2 or 3");

    if constexpr (DIM == 2) {
        static double array[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                                  1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                                  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
        return array[k];
    }

    if constexpr (DIM == 3) {
        static double array[19] = {
            1.0 / 3.0,  // k = 0
            1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
            1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,  // k = 1,..6
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,  // k = 7,..12
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0  // k = 13,..18
        };
        return array[k];
    }
}
/**
 * Update the velocity and density of the fluid after the
 * collide and stream step.
 * @param in_voxels The input lattice to use for computing
 *        the next collide and stream step from the 
 *        previous result.
 * @param old_voxels The previous output lattice from the last
 *        computation step to be updated.
 * @param mask The mask to use for identifying which
 *        boundary condition to apply to a grid cell.
 *        0 - Don't apply the collide and stream step.
 *        1 - Apply the collide and stream step.
 * @param out_velocity The ouotput velocity.
 * @param density The computed density from the voxels,
 *        boundary conditions, and velocity
 * @param density_temp The density from the voxels,
 *        used as a temporary field for reading
 *        as a stencil in the boundary conditions step.
 * @param tau The characteristic timescale.
 */
template <unsigned int DIM,
          unsigned int COMP,
          typename RealFieldT,
          typename MaskFeildT>
Neon::set::Container computeVelocity(const RealFieldT&         in_voxels,
                                     RealFieldT&               old_voxels,
                                     const MaskFeildT&         mask,
                                     RealFieldT&               out_velocity,
                                     RealFieldT&               density,
                                     RealFieldT&               density_temp,
                                     typename RealFieldT::Type tau)
{
    using T = typename RealFieldT::Type;
    return in_voxels.getGrid().getContainer(
        "ComputeVelocity", [&, tau](Neon::set::Loader& loader) {
            const auto& ins = loader.load(in_voxels);
            auto&       olds = loader.load(old_voxels);
            const auto& m = loader.load(mask);
            auto&       rho = loader.load(density);
            auto&       rho_temp = loader.load(density_temp);
            auto&       out_vel = loader.load(out_velocity);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename RealFieldT::Cell& idx) mutable {
                typename RealFieldT::ngh_idx ngh(0, 0, 0);
                const T                      default_value = 0;
                T                            r = 0;
                T                            vels[DIM];
                for (int i = 0; i < DIM; i++) {
                    vels[i] = 0.0;
                }
                int mask_val = m.nghVal(idx, ngh, 0, default_value).value;
                for (int k = 0; k < COMP; k++) {
                    T f = ins.nghVal(idx, ngh, k, default_value).value;
                    olds(idx, k) = f;
                    r += f;
                    for (int i = 0; i < DIM; i++) {
                        T e_i = get_e<DIM>(k, i);
                        vels[i] += e_i * f;
                    }
                }
                if (mask_val != 0) {
                    for (int i = 0; i < DIM; i++) {
                        vels[i] /= r;
                        out_vel(idx, i) = vels[i];
                    }
                    rho(idx, 0) = r;
                    rho_temp(idx, 0) = r;
                }
            };
        });
}


/**
 * Apply the collision and streaming step of the LBM algorithm.
 * This should be the first container in the sequence.
 * @param density The density of the fluid at each voxel.
 * @param in_velocity The velocity of the fluid from the 
 *        previous step.
 * @param in_voxels The input lattice to use for computing
 *        the next collide and stream step from the 
 *        previous result.
 * @param mask The mask to use for identifying which
 *        boundary condition to apply to a grid cell.
 *        0 - Don't apply the collide and stream step.
 *        1 - Apply the collide and stream step.
 * @param out_voxels The output lattice where the result will be
 *         stored.
 * @param tau The characteristic timescale.
 */
template <unsigned int DIM,
          unsigned int COMP,
          typename RealFieldT,
          typename MaskFeildT>
Neon::set::Container collideAndStream(const RealFieldT&         density,
                                      const RealFieldT&         in_velocity,
                                      const RealFieldT&         in_voxels,
                                      const MaskFeildT&         mask,
                                      RealFieldT&               out_voxels,
                                      typename RealFieldT::Type tau)
{
    using T = typename RealFieldT::Type;
    return in_voxels.getGrid().getContainer(
        "CollideAndStream", [&, tau](Neon::set::Loader& loader) {
            const auto& ins = loader.load(in_voxels, Neon::Compute::STENCIL);
            const auto& in_vel = loader.load(in_velocity, Neon::Compute::STENCIL);
            const auto& rho = loader.load(density, Neon::Compute::STENCIL);
            const auto& m = loader.load(mask);
            auto&       out = loader.load(out_voxels);


            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename RealFieldT::Cell& idx) mutable {
                typename RealFieldT::ngh_idx ngh(0, 0, 0);
                const T                      default_value = 0;

                int mask_val = m.nghVal(idx, ngh, 0, default_value).value;

                for (int k = 0; k < COMP; k++) {
                    T vel = 0;
                    T eu = 0;
                    T uv = 0;
                    ngh.x = -get_e<DIM>(k, 0);
                    ngh.y = -get_e<DIM>(k, 1);
                    ngh.z = -get_e<DIM>(k, 2);
                    T fold = ins.nghVal(idx, ngh, k, default_value).value;
                    T r = rho.nghVal(idx, ngh, 0, default_value).value;
                    for (int i = 0; i < DIM; i++) {
                        int e_i = get_e<DIM>(k, i);
                        vel = in_vel.nghVal(idx, ngh, i, default_value).value;
                        eu += (e_i * vel);
                        uv += (vel * vel);
                    }
                    T feq = get_w<DIM>(k) * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
                    if (mask_val != 0) {
                        out(idx, k) = ((1.0 - 1.0 / tau) * fold + 1.0 / tau * feq);
                    }
                }
            };
        });
}

/**
 * Apply boundary conditions to the lattice. This should
 * be the last container.
 * @param in_voxels The input lattice to use for computing
 *        the boundary conditions. This should be the output
 *        voxels from the previous coontainer.
 * @param in_velocity The output velocity from the previous
 *        step.
 * @param boundary_mask The mask to use for identifying which
 *        boundary condition to apply to a grid cell.
 *        0 - no boundary conditions
 *        1 - Apply fixed boundary conditions
 *        2 - Apply Neumann boundary conditions
 * @param read_density The temporary density field from the
 *        previous container's output. Used to read from as a
 *        stencil.
 * @param out_voxels The output lattice where the result will be
 *         stored.
 * @param out_velocity The ouotput velocity.
 * @param density The computed density from the voxels,
 *        boundary conditions, and velocity
 * @param tau The characteristic timescale.
 * @param dim The dimensions of the field.
 */
template <unsigned int DIM,
          unsigned int COMP,
          typename RealFieldT,
          typename MaskFeildT>
Neon::set::Container boundaryConditions(const RealFieldT&               in_voxels,
                                        const RealFieldT&               in_velocity,
                                        const MaskFeildT&               boundary_mask,
                                        const RealFieldT&               read_density,
                                        RealFieldT&                     out_voxels,
                                        RealFieldT&                     out_velocity,
                                        RealFieldT&                     density,
                                        const typename RealFieldT::Type tau,
                                        const typename RealFieldT::Type sphere_x,
                                        const typename RealFieldT::Type sphere_y)
{
    using T = typename RealFieldT::Type;
    return in_voxels.getGrid().getContainer(
        "BoundaryConditions", [&, tau](Neon::set::Loader& loader) {
            const auto& ins = loader.load(in_voxels, Neon::Compute::STENCIL);
            const auto& in_vel = loader.load(in_velocity, Neon::Compute::STENCIL);
            const auto& mask = loader.load(boundary_mask);
            const auto& rho_old = loader.load(read_density, Neon::Compute::STENCIL);
            auto&       outs = loader.load(out_voxels);
            auto&       out_vel = loader.load(out_velocity);
            auto&       rho = loader.load(density);

            T bc_values[6][3];
            if constexpr (DIM == 2) {
                const T bc_values_temp[6][3] = {
                    {0.0, 0.0, 0.0},  // left, 0
                    {0.1, 0.0, 0.0},  // top, 1 // 0.1, 0.0
                    {0.0, 0.0, 0.0},  // right, 2
                    {0.0, 0.0, 0.0},  // bottom, 3
                    {0.0, 0.0, 0.0},  // z = 0, 4
                    {0.0, 0.0, 0.0}   // z = nz - 1, 5
                };
                for (int i = 0; i < 6; i++) {
                    for (int j = 0; j < 3; j++) {
                        bc_values[i][j] = bc_values_temp[i][j];
                    }
                }
            } else {  // DIM == 3
                const T bc_values_temp[6][3] = {
                    {0.0, 0.0, 0.0},  // left, 0
                    {0.2, 0.0, 0.0},  // top, 1
                    {0.0, 0.0, 0.0},  // right, 2
                    {0.0, 0.0, 0.0},  // bottom, 3
                    {0.0, 0.0, 0.0},  // z = 0, 4
                    {0.0, 0.0, 0.0}   // z = nz - 1, 5
                };
                for (int i = 0; i < 6; i++) {
                    for (int j = 0; j < 3; j++) {
                        bc_values[i][j] = bc_values_temp[i][j];
                    }
                }
            }


            const T x_c = sphere_x;
            const T y_c = sphere_y;

            const Neon::index_3d dims = in_voxels.getDimension();

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename RealFieldT::Cell& idx) mutable {
                typename RealFieldT::ngh_idx ngh(0, 0, 0);
                const T                      default_value = 0;
                T                            vels[DIM];
                T                            new_vals[DIM];
                T                            e_i[DIM];

                int nx = dims.x, ny = dims.y;
                int boundary = mask.nghVal(idx, ngh, 0, default_value).value;
                for (int i = 0; i < DIM; i++) {
                    vels[i] = in_vel.nghVal(idx, ngh, i, default_value).value;
                }

                Neon::index_3d offsets;
                int            dr = -1;

                auto id_global = ins.mapToGlobal(idx);

                if constexpr (DIM == 2) {
                    if (id_global.x == 0) {
                        offsets.x = 1;
                        dr = 0;
                    } else if (id_global.x == dims.x - 1) {
                        offsets.x = -1;
                        dr = 2;
                    }
                    if (id_global.y == 0) {
                        offsets.y = 1;
                        dr = 1;
                    } else if (id_global.y == dims.y - 1) {
                        offsets.y = -1;
                        dr = 3;
                    }
                }

                if constexpr (DIM == 3) {
                    if (id_global.z == 0) {
                        offsets.z = 1;
                        dr = 4;
                    } else if (id_global.z == dims.z - 1) {
                        offsets.z = -1;
                        dr = 5;
                    }
                    if (id_global.x == 0) {
                        offsets.x = 1;
                        dr = 0;
                    } else if (id_global.x == dims.x - 1) {
                        offsets.x = -1;
                        dr = 2;
                    }
                    if (id_global.y == 0) {
                        offsets.y = 1;
                        dr = 1;
                    } else if (id_global.y == dims.y - 1) {
                        offsets.y = -1;
                        dr = 3;
                    }
                }


                if (dr == -1 && boundary == 1) {
                    if (id_global.x >= x_c)
                        offsets.x = 1;
                    else
                        offsets.x = -1;
                    if (id_global.y >= y_c)
                        offsets.y = 1;
                    else
                        offsets.y = -1;
                }
                ngh.x = offsets.x;
                ngh.y = offsets.y;
                ngh.z = offsets.z;

                T r = rho_old.nghVal(idx, ngh, 0, default_value).value;
                if (boundary == 1) {  // fixed boundary
                    for (int i = 0; i < DIM; i++) {
                        new_vals[i] = (dr != -1) ? bc_values[dr][i] : 0;
                        out_vel(idx, i) = new_vals[i];
                    }
                } else if (boundary == 2) {  // Neumann
                    for (int i = 0; i < DIM; i++) {
                        new_vals[i] = in_vel.nghVal(idx, ngh, i, default_value).value;
                        out_vel(idx, i) = new_vals[i];
                    }
                } else {  // we can add more types later
                    for (int i = 0; i < DIM; i++) {
                        new_vals[i] = vels[i];
                        out_vel(idx, i) = new_vals[i];
                    }
                }
                // update density, velocity
                rho(idx, 0) = r;
                for (int i = 0; i < DIM; i++) {
                    vels[i] = in_vel.nghVal(idx, ngh, i, default_value).value;
                }
                for (int k = 0; k < COMP; k++) {
                    for (int i = 0; i < DIM; i++) {
                        e_i[i] = get_e<DIM>(k, i);
                        e_i[i] = 0;
                    }
                    T fold = ins.nghVal(idx, ngh, k, default_value).value;
                    T eu = 0;
                    T uv = 0;
                    for (int i = 0; i < DIM; i++) {
                        eu += e_i[i] * new_vals[i];
                        uv += new_vals[i] * new_vals[i];
                    }
                    T vbc = get_w<2>(k) * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

                    eu = uv = 0;
                    for (int i = 0; i < DIM; i++) {
                        eu += e_i[i] * new_vals[i];
                        uv += new_vals[i] * new_vals[i];
                    }
                    T vnb = get_w<2>(k) * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
                    if (boundary != 0) {
                        // note f_eq only depends on velocity, rho
                        outs(idx, k) = vbc - vnb + fold;
                    }
                }
            };
        });
}


template <unsigned int DIM, unsigned int COMP, typename RealFieldT, typename MaskFeildT>
inline void setup(const FlowType                  flow_type,
                  MaskFeildT&                     boundary_mask,
                  MaskFeildT&                     center_mask,
                  RealFieldT&                     lattice_1,
                  RealFieldT&                     lattice_2,
                  RealFieldT&                     velocity_1,
                  RealFieldT&                     velocity_2,
                  RealFieldT&                     rho_1,
                  RealFieldT&                     rho_2,
                  const typename RealFieldT::Type sphere_x,
                  const typename RealFieldT::Type sphere_y,
                  const typename RealFieldT::Type sphere_r)
{
    auto dim = boundary_mask.getDimension();
    using T = typename RealFieldT::Type;

    if (DIM == 2) {
        if (flow_type == FlowType::border) {
            boundary_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                        idx.y < dim.y - 1) {
                        val = 1;
                    } else {
                        val = 0;
                    }
                });
            center_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                        idx.y < dim.y - 1) {
                        val = 0;
                    } else {
                        val = 1;
                    }
                });
        } else if (flow_type == FlowType::obstacle) {
            boundary_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    T dist = (idx.x - sphere_x) * (idx.x - sphere_x) +
                             (idx.y - sphere_y) * (idx.y - sphere_y);
                    bool in_circle = dist <= sphere_r * sphere_r;
                    if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                        idx.y < dim.y - 1 && !in_circle) {
                        val = 1;
                    } else {
                        val = 0;
                    }
                });
            center_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    T dist = (idx.x - sphere_x) * (idx.x - sphere_x) +
                             (idx.y - sphere_y) * (idx.y - sphere_y);
                    bool in_circle = dist <= sphere_r * sphere_r;
                    if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                        idx.y < dim.y - 1 && !in_circle) {
                        // if( idx.x > 0 && idx.x < dim.x - 1) {
                        val = 0;
                    } else if (idx.x == dim.x - 1 && idx.y > 0 &&
                               idx.y < dim.y - 1) {
                        val = 2;
                    } else {
                        val = 1;
                    }
                });
        } else {
            printf("unknown flow type during setup()");
            exit(EXIT_FAILURE);
        }
        lattice_1.forEachActiveCell(
            [&](const Neon::index_3d&, const int& c, T& val) {
                if (c == 0) {
                    val = 4.0 / 9.0;
                } else if (c >= 1 && c <= 4) {
                    val = 1.0 / 9.0;
                } else {
                    val = 1.0 / 36.0;
                }
            });
        lattice_2.forEachActiveCell(
            [&](const Neon::index_3d&, const int& c, T& val) {
                if (c == 0) {
                    val = 4.0 / 9.0;
                } else if (c >= 1 && c <= 4) {
                    val = 1.0 / 9.0;
                } else {
                    val = 1.0 / 36.0;
                }
            });
    } else if (DIM == 3) {
        boundary_mask.forEachActiveCell(
            [&](const Neon::index_3d& idx, const int& c, int& val) {
                if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                    idx.y < dim.y - 1 && idx.z > 0 && idx.z < dim.z - 1) {
                    val = 1;
                } else {
                    val = 0;
                }
            });
        center_mask.forEachActiveCell(
            [&](const Neon::index_3d& idx, const int& c, int& val) {
                if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                    idx.y < dim.y - 1 && idx.z > 0 && idx.z < dim.z - 1) {
                    val = 0;
                } else {
                    val = 1;
                }
            });
        lattice_1.forEachActiveCell(
            [&](const Neon::index_3d&, const int& c, T& val) {
                if (c == 0) {
                    val = 1.0 / 3.0;
                } else if (c >= 1 && c <= 6) {
                    val = 1.0 / 18.0;
                } else {
                    val = 1.0 / 36.0;
                }
            });
        lattice_2.forEachActiveCell(
            [&](const Neon::index_3d&, const int& c, T& val) {
                if (c == 0) {
                    val = 1.0 / 3.0;
                } else if (c >= 1 && c <= 6) {
                    val = 1.0 / 18.0;
                } else {
                    val = 1.0 / 36.0;
                }
            });
    }
    rho_1.forEachActiveCell([](const Neon::index_3d&, const int& c, T& val) { val = 1; });
    rho_2.forEachActiveCell([](const Neon::index_3d&, const int& c, T& val) { val = 1; });
    velocity_1.forEachActiveCell([](const Neon::index_3d&, const int& c, T& val) { val = 0; });
    velocity_2.forEachActiveCell([](const Neon::index_3d&, const int& c, T& val) { val = 0; });

    lattice_1.updateDeviceData(0);
    lattice_2.updateDeviceData(0);
    rho_1.updateDeviceData(0);
    rho_2.updateDeviceData(0);
    boundary_mask.updateDeviceData(0);
    center_mask.updateDeviceData(0);
    velocity_1.updateDeviceData(0);
    velocity_2.updateDeviceData(0);
}


template <unsigned int DIM, unsigned int COMP, typename RealFieldT, typename MaskFeildT>
inline void run(const int                       num_frames,
                const FlowType                  flow_type,
                MaskFeildT&                     boundary_mask,
                MaskFeildT&                     center_mask,
                RealFieldT&                     lattice_1,
                RealFieldT&                     lattice_2,
                RealFieldT&                     velocity_1,
                RealFieldT&                     velocity_2,
                RealFieldT&                     rho_1,
                RealFieldT&                     rho_2,
                const typename RealFieldT::Type tau,
                const typename RealFieldT::Type sphere_x,
                const typename RealFieldT::Type sphere_y,
                const typename RealFieldT::Type sphere_r)
{
    const auto& backend = boundary_mask.getBackend();

    Neon::skeleton::Skeleton sk(backend);
    Neon::skeleton::Options  opt;

    std::vector<Neon::set::Container> containers;


    containers.push_back(collideAndStream<DIM, COMP>(
        rho_1, velocity_1, lattice_1, boundary_mask, lattice_2, tau));

    containers.push_back(computeVelocity<DIM, COMP>(
        lattice_2, lattice_1, boundary_mask, velocity_2, rho_1, rho_2, tau));

    containers.push_back(boundaryConditions<DIM, COMP>(
        lattice_2, velocity_2, center_mask, rho_2, lattice_1, velocity_1, rho_1, tau, sphere_x, sphere_y));


    sk.sequence(containers, "LBM", opt);


    int save_id = 0;

    int t = (DIM == 2) ? 1000 : 40;
    for (int f = 0; f < num_frames; ++f) {
        sk.run();

        if (f % t == 0) {
            backend.syncAll();
            velocity_1.updateHostData(0);
            exportVTI(save_id, velocity_1);
            printf("\n frame  %d exported", f);
            save_id++;
        }
    }
    backend.syncAll();
}

int main(int argc, char** argv)
{
    Neon::init();
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {

        auto             runtime = Neon::Runtime::stream;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);

        //2D
        //constexpr int DIM = 2;
        //constexpr int COMP = 9;

        //3D
        constexpr int DIM = 3;
        constexpr int COMP = 19;


        const FlowType flow_type = FlowType::border;

        const int   dim_x = (DIM == 3) ? 64 : ((flow_type == FlowType::border) ? 256 : 801);
        const int   dim_y = (DIM == 3) ? 64 : ((flow_type == FlowType::border) ? 256 : 201);
        const int   dim_z = (DIM < 3) ? 1 : 64;
        const float niu = (DIM == 3) ? 0.063 : ((flow_type == FlowType::border) ? 0.0255f : 0.01f);
        const float tau = 3.0 * niu + 0.5;

        const float sphere_x = 160.0;
        const float sphere_y = 100.0;
        const float sphere_r = 20.0;

        const Neon::index_3d grid_dim(dim_x, dim_y, dim_z);
        const size_t         num_frames = (DIM == 2) ? 60000 : 2000;

        using Grid = Neon::dGrid;
        using dataT = float;

        Grid grid(
            backend, grid_dim, [](Neon::index_3d idx) { return true; },
            create_stencil<DIM, COMP>(), true);

        constexpr int lattice_cardinality = COMP;  // 9 or 19
        constexpr int velocity_cardinality = DIM;  // 2 or 3
        constexpr int rho_cardinality = 1;
        dataT         inactive = 0;

        auto lattice_1 = grid.template newField<dataT>("lattice_1", lattice_cardinality, inactive);
        auto lattice_2 = grid.template newField<dataT>("lattice_2", lattice_cardinality, inactive);

        auto velocity_1 = grid.template newField<dataT>("velocity_1", velocity_cardinality, inactive);
        auto velocity_2 = grid.template newField<dataT>("velocity_2", velocity_cardinality, inactive);

        auto rho_1 = grid.template newField<dataT>("rho_1", rho_cardinality, inactive);
        auto rho_2 = grid.template newField<dataT>("rho_2", rho_cardinality, inactive);


        auto boundary_mask = grid.template newField<int>("boundary_mask", rho_cardinality, int(inactive));
        auto center_mask = grid.template newField<int>("center_mask", rho_cardinality, int(inactive));

        setup<DIM, COMP>(flow_type, boundary_mask, center_mask, lattice_1, lattice_2, velocity_1, velocity_2, rho_1, rho_2, sphere_x, sphere_y, sphere_r);

        run<DIM, COMP>(num_frames, flow_type, boundary_mask, center_mask, lattice_1, lattice_2, velocity_1, velocity_2, rho_1, rho_2, tau, sphere_x, sphere_y, sphere_r);
    }
}