#pragma once


Neon::float_3d mapToCube(Neon::index_3d p, Neon::index_3d dim)
{
    //map p to an axis-aligned cube from -1 to 1
    Neon::float_3d half_dim = dim.newType<float>() * 0.5;
    Neon::float_3d ret = (p.newType<float>() - half_dim) / half_dim;
    return ret;
}

inline float sdfDodecahedron(Neon::float_3d p, float r = 1.0)
{
    //https://github.com/tovacinni/sdf-explorer/blob/master/data-files/sdf/Geometry/Dodecahedron.glsl
    const float scale = 0.7;
    p *= 1. / scale;
    float d = 0.;

    constexpr float PHI = 1.618033988749895;

    auto dot = [&](const Neon::float_3d& x, const Neon::float_3d& y) -> float {
        return x.v[0] * y.v[0] + x.v[1] * y.v[1] + x.v[2] * y.v[2];
    };

    auto normalize = [&](const Neon::float_3d& x) -> Neon::float_3d {
        float len = std::sqrt(dot(x, x));
        return x / len;
    };

    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(0.0, PHI, 1.0)))));
    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(0.0, -PHI, 1.0)))));
    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(1.0, 0.0, PHI)))));
    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(-1.0, 0.0, PHI)))));
    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(PHI, 1.0, 0.0)))));
    d = std::max(d, std::abs(dot(p, normalize(Neon::float_3d(-PHI, 1.0, 0.0)))));

    return (d - r) * scale;
}

inline float sdfMenger(Neon::float_3d p)
{
    auto mod = [](float x, float y) { return x - y * floor(x / y); };

    auto length = [&](const Neon::float_3d& x) -> float {
        return std::sqrt(x.v[0] * x.v[0] + x.v[1] * x.v[1] + x.v[2] * x.v[2]);
    };

    auto maxcomp = [](Neon::float_3d q) { return std::max(q.x, std::max(q.y, q.z)); };

    auto sdBox = [length, maxcomp](Neon::float_3d q, Neon::float_3d b) -> float {
        Neon::float_3d di(std::abs(q.x) - b.x,
                          std::abs(q.y) - b.y,
                          std::abs(q.z) - b.z);

        float mc = maxcomp(di);

        Neon::float_3d di_max(std::max(di.x, 0.f),
                              std::max(di.y, 0.f),
                              std::max(di.z, 0.f));

        return std::min(mc, length(di_max));
    };

    Neon::float_3d one(1.f, 1.f, 1.f);

    float d = sdBox(p, one);


    float s = 1.0;
    for (int m = 0; m < 4; m++) {

        Neon::float_3d a(mod(p.x * s, 2.f) - 1.f,
                         mod(p.y * s, 2.f) - 1.f,
                         mod(p.z * s, 2.f) - 1.f);

        s *= 3.0;

        Neon::float_3d r(std::abs(1.0 - 3.0 * std::abs(a.x)),
                         std::abs(1.0 - 3.0 * std::abs(a.y)),
                         std::abs(1.0 - 3.0 * std::abs(a.z)));

        float da = std::max(r.x, r.y);
        float db = std::max(r.y, r.z);
        float dc = std::max(r.z, r.x);
        float c = (std::min(da, std::min(db, dc)) - 1.0) / s;

        d = std::max(d, c);
    }

    return d;
}

void MultiResDemo()
{
    int              nGPUs = 1;
    Neon::int32_3d   dim(128, 128, 128);
    std::vector<int> gpusIds(nGPUs, 0);
    auto             bk = Neon::Backend(gpusIds, Neon::Runtime::stream);

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1, 1, 1});

    const float eps = std::numeric_limits<float>::epsilon();

    Neon::domain::mGrid grid(
        bk,
        dim,
        {[&](const Neon::index_3d id) -> bool {
             //return sdfDodecahedron(mapToCube(id, dim)) < eps;
             return sdfMenger(mapToCube(id, dim)) < eps;
         },
         [&](const Neon::index_3d&) -> bool {
             return false;
         },
         [&](const Neon::index_3d&) -> bool {
             return false;
         },
         [&](const Neon::index_3d&) -> bool {
             return false;
         },
         [&](const Neon::index_3d&) -> bool {
             return false;
         }},
        Neon::domain::Stencil::s7_Laplace_t(),
        descriptor);

    std::stringstream s("mGridDemo", std::ios_base::app | std::ios_base::out);

    for (int i = 0; i < descriptor.getDepth(); ++i) {
        s << descriptor.getLog2RefFactor(i);
    }

    grid.topologyToVTK(s.str() + ".vtk", false);

    auto field = grid.newField<float>("myField", 1, -10000);

    for (int l = 0; l < descriptor.getDepth(); ++l) {
        field.forEachActiveCell(
            l,
            [&]([[maybe_unused]] const Neon::int32_3d idx, const int, float& val) {
                if (l == 0) {
                    val = mapToCube(idx, dim).norm();
                } else {
                    val = 0;
                }
            });
    }

    field.updateCompute();

    for (int level = 1; level < descriptor.getDepth(); ++level) {

        auto container = grid.getContainer(
            "container", level, [&, level](Neon::set::Loader& loader) {
                auto& local = field.load(loader, level, Neon::MultiResCompute::MAP);
                return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                    if (!local.hasChildren(cell)) {

                        local(cell, 0) = -1.0;

                    } else {

                        float     val = 0;
                        const int refFactor = local.getRefFactor(level - 1);

                        for (int8_t z = 0; z < refFactor; ++z) {
                            for (int8_t y = 0; y < refFactor; ++y) {
                                for (int8_t x = 0; x < refFactor; ++x) {

                                    Neon::int8_3d child_dir(x, y, z);

                                    val = std::max(val, local.childVal(cell, child_dir, 0, 0).value);
                                }
                            }
                            local(cell, 0) = val;
                        }
                    }
                };
            });

        container.run(0);
        grid.getBackend().syncAll();
    }


    field.updateIO();

    field.ioToVtk(s.str(), "f");
}


TEST(MultiRes, Demo)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResDemo();
    }
}