// References
// 2D LBM: https://github.com/hietwll/LBM_Taichi
// 2D LBM Verification data:
// https://www.sciencedirect.com/science/article/pii/0021999182900584 For 2D/3D
// constants: https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods

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
    return Neon::domain::Stencil::s19_t(false);  // filterCenterOut = false;
}

template <unsigned int DIM, typename T, typename Grid, typename Field>
inline void exportVTI(const int t, Field& field, Grid& grid)
{
    printf("\n Exporting Frame =%d", t);
    int                precision = 4;
    std::ostringstream oss;
    oss << std::setw(precision) << std::setfill('0') << t;
    std::string prefix = "lbm" + std::to_string(DIM) + "D_";
    std::string fname = prefix + oss.str();
    field.ioToVtk(fname, "field");
}

template <unsigned int DIM, unsigned int COMP, typename RealFieldT, typename MaskFeildT>
inline void setup(const FlowType flow_type,
                  MaskFeildT&    boundary_mask,
                  MaskFeildT&    center_mask,
                  RealFieldT&    lattice_1,
                  RealFieldT&    lattice_2,
                  RealFieldT&    velocity_1,
                  RealFieldT&    velocity_2,
                  RealFieldT&    rho_1,
                  RealFieldT&    rho_2)
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
            T x_c = 160.0;
            T y_c = 100.0;
            T r_c = 20.0;
            boundary_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    T dist = (idx.x - x_c) * (idx.x - x_c) +
                             (idx.y - y_c) * (idx.y - y_c);
                    bool in_circle = dist <= r_c * r_c;
                    if (idx.x > 0 && idx.x < dim.x - 1 && idx.y > 0 &&
                        idx.y < dim.y - 1 && !in_circle) {
                        val = 1;
                    } else {
                        val = 0;
                    }
                });
            center_mask.forEachActiveCell(
                [&](const Neon::index_3d& idx, const int& c, int& val) {
                    T dist = (idx.x - x_c) * (idx.x - x_c) +
                             (idx.y - y_c) * (idx.y - y_c);
                    bool in_circle = dist <= r_c * r_c;
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

    lattice_1.updateCompute(0);
    lattice_2.updateCompute(0);
    rho_1.updateCompute(0);
    rho_2.updateCompute(0);
    boundary_mask.updateCompute(0);
    center_mask.updateCompute(0);
    velocity_1.updateCompute(0);
    velocity_2.updateCompute(0);
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

        const Neon::index_3d grid_dim(dim_x, dim_y, dim_z);
        const size_t         num_frames = (DIM == 2) ? 60000 : 2000;

        using Grid = Neon::domain::dGrid;
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

        setup<DIM, COMP>(flow_type, boundary_mask, center_mask, lattice_1, lattice_2, velocity_1, velocity_2, rho_1, rho_2);
    }
}