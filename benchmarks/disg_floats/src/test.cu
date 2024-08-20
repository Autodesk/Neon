#include "Config.h"

#include "Neon/domain/Grids.h"
#include "Neon/domain/aGrid.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/set/memory/memSet.h"
#include "Repoert.h"

// #include <fenv.h>
// #include "/usr/include/fenv.h"

union D
{
    double d;
    float  f[2];

    inline NEON_CUDA_HOST_DEVICE D() {}

    inline NEON_CUDA_HOST_DEVICE D(float const& f0, float const& f1)
    {
        setFloat(f0, f1);
    }

    inline NEON_CUDA_HOST_DEVICE D(double const& d0)
    {
        setDouble(d0);
    }

    inline auto NEON_CUDA_HOST_DEVICE operator()() -> double
    {
        return d;
    }

    inline auto NEON_CUDA_HOST_DEVICE operator()(float const& f0, float const& f1)
    {
        setFloat(f0, f1);
    }

    inline auto NEON_CUDA_HOST_DEVICE operator()(double const& dd)
    {
        setDouble(dd);
    }

    inline auto NEON_CUDA_HOST_DEVICE setFloat(float const& f0, float const& f1) -> void
    {
        f[0] = f0;
        f[1] = f1;
    }

    inline auto NEON_CUDA_HOST_DEVICE setDouble(double const& dd) -> void
    {
        d = dd;
    }

    template <int component>
    inline auto NEON_CUDA_HOST_DEVICE getFloat() -> float
    {
        if constexpr (component == 0) {
            return f[0];
        }
        if constexpr (component == 1) {
            return f[1];
        }
    }

    inline auto NEON_CUDA_HOST_DEVICE getDouble() -> float
    {
        return d;
    }
};

NEON_CUDA_KERNEL void axpy_classic(double const* __restrict__ x,
                                   double* __restrict__ y,
                                   double a,
                                   int    n,
                                   int    cardinality)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        for (int c = 0; c < cardinality; c++) {
            y[i] += a * x[i];
            i += n;
        }
    }
}

NEON_CUDA_KERNEL void axpy_diss(double const* __restrict__ x,
                                double* __restrict__ y,
                                double a,
                                int    n,
                                int    cardinality)
{
    float const* __restrict__ xf = (float const*)x;
    float* __restrict__ yf = (float*)y;

    unsigned int i1 = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i2 = i1 + n;
    const int    twice_n = 2 * n;
    if (i1 < n) {

        for (int c = 0; c < cardinality; c++) {
            register D y, x;
            x.f[0] = xf[i1];
            x.f[1] = xf[i2];

            y.f[0] = yf[i1];
            y.f[1] = yf[i2];

            y.d = y.d + a * x.d;

            yf[i1] = y.f[0];
            yf[i2] = y.f[1];

            i1 += 2 * n;
            i2 += 2 * n;
        }
    }
}

NEON_CUDA_KERNEL void from_classic_to_diss(double const* xf, double* y, int n, int cardinality)
{

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i1 = i;
    unsigned int i2 = i + n;

    if (i < n) {
        for (int c = 0; c < cardinality; c++) {
            float* yf = (float*)y;

            double const read = xf[i];
            D            dy(read);
            yf[i1] = dy.getFloat<0>();
            yf[i2] = dy.getFloat<1>();
            i += +n;
            i1 += 2 * n;
            i2 += 2 * n;
        }
    }
}

NEON_CUDA_KERNEL void from_diss_to_classic(double const* y, double* x, int n, int cardinality)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i1 = i;
    unsigned int i2 = i + n;

    if (i < n) {
        float const* yf = (float const*)y;
        for (int c = 0; c < cardinality; c++) {

            D      dy(yf[i1], yf[i2]);
            double read = dy();
            x[i] = read;
            i += +n;
            i1 += 2 * n;
            i2 += 2 * n;
        }
    }
}

namespace details {

struct Test
{
    using Grid = Neon::dGrid;
    using Field = Grid::Field<double, 0>;

    Neon::Backend bk;
    Grid          grid;
    Config        config;
    double        contantReal = 2.0;

    Test(Config& config)
    {
        this->config = config;
        Neon::init();
        bk = Neon::Backend(1, Neon::Runtime::stream);
        std::cout << "Backend: " << bk.toString() << std::endl;
        grid = Grid(bk, Neon::int32_3d(config.n, 1, 1), [](Neon::index_3d& idx) { return true; }, Neon::domain::Stencil::s6_Jacobi_t());
        std::cout << "Grid: " << grid.toString() << std::endl;
    }

    auto reset_host(Field& field, int offset = 0)
    {
        field.forEachActiveCell(
            [offset, &field](Neon::index_3d const& idx, std::vector<double*>& a) -> void {
                for (int c = 0; c < field.getCardinality(); c++) {
                    *(a[c]) = idx.x + idx.y + idx.z + c + double(offset);
                }
            });
        bk.sync(0);
        return field;
    }

    auto getField(int offset = 0)
    {
        auto field = grid.newField<double, 0>("a", config.cardinality, 0);
        field.forEachActiveCell(
            [offset, &field](Neon::index_3d const& idx, std::vector<double*>& a) -> void {
                for (int c = 0; c < field.getCardinality(); c++) {
                    *(a[c]) = idx.x + idx.y + idx.z + c + double(offset);
                }
            });
        field.updateDeviceData(0);
        bk.sync(0);
        this->reset_host(field, 33);
        return field;
    }


    auto getMem(Field& field, Neon::Execution execution) -> double*
    {
        return field.getPartition(execution, 0, Neon::DataView::STANDARD).mem();
    }

    auto getMemConst(Field& field, Neon::Execution execution) -> double const*
    {
        return field.getPartition(execution, 0, Neon::DataView::STANDARD).mem();
    }

    auto axpyContainer(Field& X, Field& Y, double a) -> void
    {
        if (config.cardinality == 1) {
            auto c = grid.newContainer(
                "axpy",
                [&](Neon::set::Loader& loader) {
                    auto const x = loader.load(X);
                    auto       y = loader.load(Y);
                    return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                        y(e, 0) += a * x(e, 0);
                    };
                });
            c.run(0);
        }
        if (config.cardinality != 1) {
            auto c = grid.newContainer(
                "axpy",
                [&](Neon::set::Loader& loader) {
                    auto const x = loader.load(X);
                    auto       y = loader.load(Y);
                    return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                        for (int c = 0; c < y.cardinality(); c++) {
                            y(e, c) += a * x(e, c);
                        }
                    };
                });
            c.run(0);
        }
        Y.updateHostData(0);
        bk.sync(0);
    }

    template <typename... Types>
    auto launch(int iterations, void (*foo)(Types...), Types... args) -> float
    {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        std::vector<void*> parameters = {&args...};
        auto&              stream = bk.streamSet(0)[0];


        auto s = bk.streamSet(0)[0].stream();
        cudaEventRecord(start, s);
        Neon::sys::GpuLaunchInfo info(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode,
                                      int64_t(config.n),
                                      config.blockSize,
                                      0);
        for (int iter = 0; iter < iterations; iter++) {
            bk.devSet().gpuDev(0).kernel.cudaLaunchKernel<Neon::run_et::async>(
                stream,
                info,
                (void*)foo,
                parameters.data());
        }
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }

    auto run_axpy()
    {
        // 1. Get fields
        auto Xclassic = getField(0);
        auto Yclassic = getField(3);
        auto Xdiss = getField(33);
        auto Ydiss = getField(33);
        auto YdissResClassicLayout = getField(33);
        auto XdissResClassicLayout = getField(33);


        bk.sync(0);
        // 3. Convert data fields to diss
        launch(1,
               from_classic_to_diss,
               getMemConst(Xclassic, Neon::Execution::device),
               getMem(Xdiss, Neon::Execution::device),
               config.n,
               config.cardinality);

        launch(1,
               from_classic_to_diss,
               getMemConst(Yclassic, Neon::Execution::device),
               getMem(Ydiss, Neon::Execution::device),
               config.n,
               config.cardinality);

        // 3. Run axpy with diss

        // 2. Run axpy with classic
        launch(1,
               axpy_classic,
               getMemConst(Xclassic, Neon::Execution::device),
               getMem(Yclassic, Neon::Execution::device),
               contantReal,
               config.n,
               config.cardinality);

        //
        launch(1,
               axpy_diss,
               getMemConst(Xdiss, Neon::Execution::device),
               getMem(Ydiss, Neon::Execution::device),
               contantReal,
               config.n,
               config.cardinality);
        // 4. Get result back to classic layout


        launch(1,
               from_diss_to_classic,
               getMemConst(Ydiss, Neon::Execution::device),
               getMem(YdissResClassicLayout, Neon::Execution::device),
               config.n,
               config.cardinality);
        launch(1,
               from_diss_to_classic,
               getMemConst(Xdiss, Neon::Execution::device),
               getMem(XdissResClassicLayout, Neon::Execution::device),
               config.n,
               config.cardinality);
        // 4. Compare results
        Yclassic.updateHostData(0);
        Xclassic.updateHostData(0);

        YdissResClassicLayout.updateHostData(0);
        XdissResClassicLayout.updateHostData(0);
        bk.sync(0);

        volatile bool x_correct = true;
        volatile bool y_correct = true;

        Yclassic.getGrid().newContainer<Neon::Execution::host>(
                              "Compare",
                              [&](Neon::set::Loader& loader) {
                                  auto const y1 = loader.load(Yclassic);
                                  auto const y2 = loader.load(YdissResClassicLayout);
                                  auto const x1 = loader.load(Xclassic);
                                  auto const x2 = loader.load(XdissResClassicLayout);
                                  return [=, &x_correct, &y_correct](const typename Field::Idx& e) mutable -> void {
                                      for (int c = 0; c < y1.cardinality(); c++) {
                                          if (y1(e, c) != y2(e, c)) {
                                              y_correct = false;
                                          }
                                          if (x1(e, c) != x2(e, c)) {
                                              x_correct = false;
                                          }
                                      }
                                  };
                              })
            .run(0);

        bk.sync();

        // Yclassic.ioToVtk("Yclassic", "Yclassic", false);
        // YdissResClassicLayout.ioToVtk("YdissResClassicLayout", "YdissResClassicLayout", false);

        // Xclassic.ioToVtk("Xclassic", "Xclassic", false);
        // XdissResClassicLayout.ioToVtk("XdissResClassicLayout", "XdissResClassicLayout", false);

        if (!(x_correct && y_correct)) {
            Neon::NeonException error("axpy");
            error << "Wrong output.";
            NEON_THROW(error);
        }
        // 5. Update report with the performance results
        // 2. Run axpy with classic
        float classic_time = launch(config.iterations,
                                    axpy_classic,
                                    getMemConst(Xclassic, Neon::Execution::device),
                                    getMem(Yclassic, Neon::Execution::device),
                                    contantReal,
                                    config.n,
                                    config.cardinality);
        std::cout << "axpy_classic time (ms): " << classic_time / config.iterations << std::endl;

        float disg_time = launch(config.iterations,
                                 axpy_diss,
                                 getMemConst(Xdiss, Neon::Execution::device),
                                 getMem(Ydiss, Neon::Execution::device),
                                 contantReal,
                                 config.n,
                                 config.cardinality);
        std::cout << "axpy_diss    time (ms): " << disg_time / config.iterations << std::endl;
    }
};

}  // namespace details

auto run(Config& config,
         Report& report,
         std::stringstream&) -> void
{
    using namespace details;
    Test test(config);
    test.run_axpy();
}