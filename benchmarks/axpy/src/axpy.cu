#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include <nvtx3/nvToolsExt.h>

template<typename Field>
auto init_data(Field &x, Field &y)
    -> Neon::set::Container {
    const auto &grid = x.getGrid();
    return grid.newContainer(
        "init_data",
        [&](Neon::set::Loader &loader) {
            auto &xd = loader.load(x);
            auto &yd = loader.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx &idx) mutable {
                for (int c = 0; c < Field::Cardinality; ++c) {
                    xd(idx, c) = 2.2;
                    yd(idx, c) = 1.0;
                }
                return;
            };
        });
}

template<typename Field>
auto axpy(Field& x, Field& y)
    -> Neon::set::Container {
    return x.getGrid().newContainer(
        "init_data",
        [&](Neon::set::Loader &loader)-> auto {
            typename Field::Type alpha = 2;
            const auto &xd = loader.load(x);
            auto &yd = loader.load(y);
            return [=] NEON_CUDA_HOST_DEVICE(
                const typename Field::Grid::Idx &idx) mutable -> void {
                for (int c = 0; c < Field::Cardinality; ++c) {
                    yd(idx, c) += xd(idx, c) +
                            alpha * yd(idx, c);
                }
            };
        });
}

template<typename dType, int cardinality>
auto axpy_repetition(Neon::index_3d dim,
                     int ngpus,
                     int iterations,
                     int repetitions, int argc, char **argv) -> void {
    std::string dtype_name;
    if (std::is_same_v<dType, float>) {
        dtype_name = "float";
    } else if (std::is_same_v<dType, double>) {
        dtype_name = "double";
    } else if (std::is_same_v<dType, int>) {
        dtype_name = "int";
    } else {
        NEON_THROW_UNSUPPORTED_OPTION("dType not supported");
    }

    // Construct the report name
    std::ostringstream report_name_stream;
    report_name_stream << "axpy_cpp_neon_" << dim.x << "_" << ngpus << "_"
            << dtype_name << "_" << iterations << "_"
            << repetitions;

    std::string report_name = report_name_stream.str();


    std::vector<double> mlups_vec;
    std::vector<double> t_vec;

    for (int rep = 0; rep < repetitions; ++rep) {
        // allocate grid and 2 fields
        Neon::Backend backend(ngpus, Neon::Runtime::stream);
        using Grid = Neon::dGrid;
        Grid grid(
            backend,
            dim,
            [](Neon::index_3d idx) { return true; },
            Neon::domain::Stencil::s6_Jacobi_t(),
            true);

        auto X = grid.template newField<dType, cardinality>("x", cardinality, 0);
        auto Y = grid.template newField<dType, cardinality>("y", cardinality, 0);

        // init data
        init_data(X, Y).run(0);

        // loop for AXPY
        auto odd = axpy(X, Y);
        auto even = axpy(Y, X);
        backend.syncAll();

        for (int i = 0; i < 4; i++) {
            odd.run(0);
            even.run(0);
        }

        backend.syncAll();
        nvtxRangePush("iterations");

        Neon::TimerMS timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            odd.run(0);
            even.run(0);
        }
        backend.syncAll();
        nvtxRangePop();

        timer.stop();
        double t = timer.time();
        t_vec.push_back(t);
        double mlups = (dim.x * dim.y * dim.z * iterations * 1.0) / t / 1000.0;
        mlups_vec.push_back(mlups);
        std::cout << "MLUPS: " << mlups << std::endl;
    }
    Neon::Report report(report_name);

    report.commandLine(argc, argv);
    report.addMember("dim", dim.x);
    report.addMember("ngpus", ngpus);
    report.addMember("dtype", dtype_name);
    report.addMember("iterations", iterations);
    report.addMember("repetitions", repetitions);
    report.addMember("runtime", "cpp_neon");

    report.addMember("mlups", mlups_vec);
    report.addMember("t", t_vec);

    report.write(report_name);
}

// Explicit instantiation of axpy_repetition

template auto axpy_repetition<int, 1>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<int, 3>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<int, 5>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<int, 19>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<int, 27>(Neon::index_3d, int, int, int, int, char **) -> void;

template auto axpy_repetition<float, 1>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<float, 3>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<float, 5>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<float, 19>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<float, 27>(Neon::index_3d, int, int, int, int, char **) -> void;

template auto axpy_repetition<double, 1>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<double, 3>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<double, 5>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<double, 19>(Neon::index_3d, int, int, int, int, char **) -> void;
template auto axpy_repetition<double, 27>(Neon::index_3d, int, int, int, int, char **) -> void;
