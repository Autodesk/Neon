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

    NEON_CUDA_HOST_DEVICE D() {}

    NEON_CUDA_HOST_DEVICE D(float const& f0, float const& f1)
    {
        setFloat(f0, f1);
    }

    NEON_CUDA_HOST_DEVICE D(double const& d0)
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


NEON_CUDA_KERNEL void axpy_classic(double* x, double* y, double a, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] += a * x[i];
    }
}

NEON_CUDA_KERNEL void axpy_diss(float* xf, float* yf, double a, int nBy2)
{
    int const i1 = threadIdx.x + blockIdx.x * 2 * blockDim.x;
    int const i2 = i1 + blockDim.x;
    if (i2 < nBy2) {
        D x(xf[i1], xf[i2]);
        D y(yf[i1], yf[i2]);
        y(y() + a * x());
        yf[i1] = y.getFloat<0>();
        yf[i2] = y.getFloat<1>();
    }
}

NEON_CUDA_KERNEL void from_classic_to_diss(double* xf, float* yf, int n)
{
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const i1 = threadIdx.x + blockIdx.x * 2 * blockDim.x;
    int const i2 = i1 + blockDim.x;
    if (i < n) {
        double read = xf[i];
        D      y(read);
        yf[i1] = y.getFloat<0>();
        yf[i2] = y.getFloat<1>();
    }
}

NEON_CUDA_KERNEL void from_diss_to_classic(float* yf, double* xf, int n)
{
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const i1 = threadIdx.x + blockIdx.x * 2 * blockDim.x;
    int const i2 = i1 + blockDim.x;
    if (i < n) {
        D      y(yf[i1], yf[i2]);
        double read = y();
        xf[i] = read;
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
    double        contantReal = 1.0;

    Test(Config& config)
    {
        this->config = config;
        Neon::init();
        bk = Neon::Backend(1, Neon::Runtime::stream);
        grid = Grid(bk, Neon::int32_3d(config.n, 1, 1), [](Neon::index_3d& idx) { return true; }, Neon::domain::Stencil::s6_Jacobi_t());
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
        return field;
    }

    auto getMem(Field& field, Neon::Execution execution) -> double*
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
    auto launch(void* foo, Types... args)
    {
        std::vector<void*>       parameters = {&args...};
        auto&                    stream = bk.streamSet(0)[0];
        Neon::sys::GpuLaunchInfo info(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode,
                                      int64_t(config.n),
                                      config.blockSize,
                                      0);
        bk.devSet().gpuDev(0).kernel.cudaLaunchKernel<Neon::run_et::async>(
            stream,
            info,
            (void*)axpy_classic,
            parameters.data());
    }

    auto run_axpy()
    {
        // 1. Get fields
        auto Xclassic = getField(0);
        auto Yclassic = getField(1);
        auto Xdiss = getField(1);
        auto Ydiss = getField(1);

        // 2. Run axpy with classic
        launch((void*)(axpy_classic),
               getMem(Xclassic, Neon::Execution::device),
               getMem(Yclassic, Neon::Execution::device),
               contantReal,
               config.n);

        bk.sync(0);
        // 3. Convert data fields

        // 3. Run axpy with diss
        // 4. Compare results
        // 5. Update report with the performance results
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