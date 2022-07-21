#include <sstream>

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

// template <typename FieldT>
// inline void draw_pixels(const int t, FieldT& field)
//{
//     printf("\n Exporting Frame =%d", t);
//     int                precision = 4;
//     std::ostringstream oss;
//     oss << std::setw(precision) << std::setfill('0') << t;
//     std::string fname = "frame_" + oss.str();
//     field.ioToVtk(fname, "pixels");
// }
//
// NEON_CUDA_HOST_DEVICE inline Neon::float_2d complex_sqr(Neon::float_2d& z)
//{
//     return Neon::float_2d(z.x * z.x - z.y * z.y, z.x * z.y * 2.0f);
// }
//
// NEON_CUDA_HOST_DEVICE inline Neon::float_2d complex_pow(Neon::float_2d& z, Neon::float_1d& n)
//{
//     Neon::float_1d radius = pow(z.norm(), n);
//     Neon::float_1d angle = n * atan2(z.y, z.x);
//     return Neon::float_2d(radius * cos(angle), radius * sin(angle));
// }
//
// template <typename FieldT>
// inline Neon::set::Container FractalsContainer(FieldT&  pixels,
//                                               int32_t& time,
//                                               int32_t  n)
//{
//     return pixels.getGrid().getContainer(
//         "FractalContainer", [&, n](Neon::set::Loader& L) {
//             auto& px = L.load(pixels);
//             auto& t = time;
//
//             return [=] NEON_CUDA_HOST_DEVICE(
//                        const typename FieldT::Cell& idx) mutable {
//                 auto id = px.mapToGlobal(idx);
//
//                 Neon::float_2d c(-0.8, cos(t * 0.03) * 0.2);
//                 Neon::float_2d z((float(id.x) / float(n)) - 1.0f,
//                                  (float(id.y) / float(n)) - 0.5f);
//                 z *= 2.0f;
//                 float iterations = 0;
//                 while (z.norm() < 20 && iterations < 50) {
//                     z = complex_sqr(z) + c;
//                     iterations += 1;
//                 }
//                 px(idx, 0) = 1.0f - iterations * 0.02;
//             };
//         });
// }

auto sdfCenteredSphere(Neon::int32_3d idx /**< queried location*/,
                       Neon::index_3d dim,
                       double         voxelEdge,
                       double         r)
    -> double
{
    auto sphereCenter = (dim.newType<double>() * voxelEdge) / 2.0;
    auto queryPoint = idx.newType<double>() * voxelEdge;
    auto diff = queryPoint - sphereCenter;
    // distance from sphere origin origin
    double d = std::pow(diff.x, 2) +
               std::pow(diff.y, 2) +
               std::pow(diff.z, 2);
    return d - r;
}


int main(int, char**)
{
    Neon::init();
    const int32_t    n = 25;
    const double     voxelEdge = 1.0;
    Neon::index_3d   dim(n, n, n);
    std::vector<int> gpu_ids{0};
    const double     r = (n * voxelEdge / 2) * .8;

    auto runtime = Neon::Runtime::stream;
    // auto runtime = Neon::Runtime::openmp;

    Neon::Backend backend(gpu_ids, runtime);

    using Grid = Neon::domain::eGrid;

    std::vector<Neon::index_3d> points{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    Neon::domain::Stencil       myStencil(points);

    Grid grid(
        backend, dim,
        [&](const Neon::index_3d& ) -> bool {
            return true;
        },
        myStencil);

    grid.ioDomainToVtk("domain");

    auto sdf = grid.newField<double>("sdf", 1, -100);

    sdf.forEachActiveCell([&](const Neon::index_3d& idx, int, double& value) {
        double sdf = sdfCenteredSphere(idx, dim, voxelEdge, r);
        value = sdf;
    });

    sdf.ioToVtk("sdf", "sdf");
    //
    //    int   cardinality = 1;
    //    float inactiveValue = 0.0f;
    //    auto  pixels = grid.template newField<float>("pixels", cardinality, inactiveValue);
    //
    //    Neon::skeleton::Skeleton skeleton(backend);
    //
    //    int32_t time;
    //    skeleton.sequence({FractalsContainer(pixels, time, n)}, "fractal");
    //
    //
    //    for (time = 0; time < 1000; ++time) {
    //        skeleton.run();
    //
    //        pixels.updateIO(0);
    //        // draw_pixels(time, pixels);
    //    }
}