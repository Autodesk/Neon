#include <iomanip>
#include <sstream>

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Field>
inline void draw_pixels(const int t, Field& field)
{
    printf("\n Exporting Frame =%d", t);
    int                precision = 4;
    std::ostringstream oss;
    oss << std::setw(precision) << std::setfill('0') << t;
    std::string fname = "frame_" + oss.str();
    field.ioToVtk(fname, "pixels");
}

NEON_CUDA_HOST_DEVICE inline Neon::float_2d complex_sqr(Neon::float_2d& z)
{
    return Neon::float_2d(z.x * z.x - z.y * z.y, z.x * z.y * 2.0f);
}

NEON_CUDA_HOST_DEVICE inline Neon::float_2d complex_pow(Neon::float_2d& z, Neon::float_1d& n)
{
    Neon::float_1d radius = pow(z.norm(), n);
    Neon::float_1d angle = n * atan2(z.y, z.x);
    return Neon::float_2d(radius * cos(angle), radius * sin(angle));
}

template <typename Field>
inline Neon::set::Container FractalsContainer(Field&  pixels,
                                              int32_t& time,
                                              int32_t  n)
{
    return pixels.getGrid().newContainer(
        "FractalContainer", [&, n](Neon::set::Loader& L) {
            auto& px = L.load(pixels);
            auto& t = time;

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Idx& idx) mutable {
                auto id = px.getGlobalIndex(idx);

                Neon::float_2d c(-0.8, cos(t * 0.03) * 0.2);
                Neon::float_2d z((float(id.x) / float(n)) - 1.0f,
                                 (float(id.y) / float(n)) - 0.5f);
                z *= 2.0f;
                float iterations = 0;
                while (z.norm() < 20 && iterations < 50) {
                    z = complex_sqr(z) + c;
                    iterations += 1;
                }
                px(idx, 0) = 1.0f - iterations * 0.02;
            };
        });
}

int main(int argc, char** argv)
{
    Neon::init();
    if ( Neon::Backend::countAvailableGpus() > 0) {
        int32_t          n = 320;
        Neon::index_3d   dim(2 * n, n, 1);
        std::vector<int> gpu_ids{0};

        auto runtime = Neon::Runtime::stream;

        //runtime = Neon::Runtime::openmp;

        Neon::Backend backend(gpu_ids, runtime);

        using Grid = Neon::dGrid;
        Grid grid(
            backend, dim,
            [](const Neon::index_3d& idx) -> bool { return true; },
            Neon::domain::Stencil::s7_Laplace_t());

        int   cardinality = 1;
        float inactiveValue = 0.0f;
        auto  pixels = grid.template newField<float>("pixels", cardinality, inactiveValue);

        Neon::skeleton::Skeleton skeleton(backend);

        int32_t time;
        skeleton.sequence({FractalsContainer(pixels, time, n)}, "fractal");


        for (time = 0; time < 1000; ++time) {
            skeleton.run();

            pixels.updateHostData(0);
            draw_pixels(time, pixels);
        }
    }
}