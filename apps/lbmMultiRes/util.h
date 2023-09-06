#pragma once
#include "Neon/Neon.h"
#include "lattice.h"


#define GLM_FORCE_SWIZZLE
#include <glm/ext.hpp>
#include <glm/glm.hpp>

constexpr NEON_CUDA_HOST_DEVICE Neon::int8_3d getDir(const int8_t q)
{
    return Neon::int8_3d(latticeVelocity[q][0], latticeVelocity[q][1], latticeVelocity[q][2]);
}

template <typename T>
constexpr NEON_CUDA_HOST_DEVICE inline Neon::int8_3d uncleOffset(const T& cell, const Neon::int8_3d& q)
{
    //given a local index within a cell and a population direction (q)
    //find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
    //this offset is wrt the cell containing the localID (i.e., the parent of localID)
    auto off = [](const int8_t i, const int8_t j) {
        //0, -1 --> -1
        //1, -1 --> 0
        //0, 0 --> 0
        //0, 1 --> 0
        //1, 1 --> 1
        const int8_t s = i + j;
        return (s <= 0) ? s : s - 1;
    };
    Neon::int8_3d offset(off(cell.x % Neon::domain::details::mGrid::kUserBlockSizeX, q.x),
                         off(cell.y % Neon::domain::details::mGrid::kUserBlockSizeY, q.y),
                         off(cell.z % Neon::domain::details::mGrid::kUserBlockSizeZ, q.z));
    return offset;
}

template <typename T>
NEON_CUDA_HOST_DEVICE T computeOmega(T omega0, int level, int numLevels)
{
    int ilevel = numLevels - level - 1;
    // scalbln(1.0, x) = 2^x
    return 2 * omega0 / (scalbln(1.0, ilevel + 1) + (1. - scalbln(1.0, ilevel)) * omega0);
}

template <typename T, int Q>
NEON_CUDA_HOST_DEVICE Neon::Vec_3d<T> velocity(const T* fin,
                                               const T  rho)
{
    Neon::Vec_3d<T> vel(0, 0, 0);
    for (int i = 0; i < Q; ++i) {
        const T f = fin[i];
        for (int d = 0; d < 3; ++d) {
            vel.v[d] += f * latticeVelocity[i][d];
        }
    }
    for (int d = 0; d < 3; ++d) {
        vel.v[d] /= rho;
    }
    return vel;
}


inline float sdfCube(Neon::index_3d id, Neon::index_3d dim, Neon::float_3d b = {1.0, 1.0, 1.0})
{
    auto mapToCube = [&](Neon::index_3d id) {
        //map p to an axis-aligned cube from -1 to 1
        Neon::float_3d half_dim = dim.newType<float>() * 0.5;
        Neon::float_3d ret = (id.newType<float>() - half_dim) / half_dim;
        return ret;
    };
    Neon::float_3d p = mapToCube(id);

    Neon::float_3d d(std::abs(p.x) - b.x, std::abs(p.y) - b.y, std::abs(p.z) - b.z);

    Neon::float_3d d_max(std::max(d.x, 0.f), std::max(d.y, 0.f), std::max(d.z, 0.f));
    float          len = std::sqrt(d_max.x * d_max.x + d_max.y * d_max.y + d_max.z * d_max.z);
    float          val = std::min(std::max(d.x, std::max(d.y, d.z)), 0.f) + len;
    return val;
}


NEON_CUDA_HOST_DEVICE inline float sdfJetfighter(glm::ivec3 id, glm::ivec3 dim)
{
    float     turn = 0.f;
    float     pitch = 0.f + glm::pi<float>();
    float     roll = 0.f;
    float     rudderAngle = 0.f;
    float     speed = 0.5;
    glm::vec3 checkPos = glm::vec3(0.f);
    glm::vec3 planePos = glm::vec3(0.f);
    float     winDist = 10000.0;
    float     engineDist = 10000.0;
    float     eFlameDist = 10000.0;
    float     blackDist = 10000.0;
    float     bombDist = 10000.0;
    float     bombDist2 = 10000.0;
    float     missileDist = 10000.0;
    float     frontWingDist = 10000.0;
    float     rearWingDist = 10000.0;
    float     topWingDist = 10000.0;
    glm::vec2 missilesLaunched = glm::vec2(0.f);

    //https://github.com/tovacinni/sdf-explorer/blob/master/data-files/sdf/Vehicle/Jetfighter.glsl
    auto mapToCube = [&](glm::ivec3 id) {
        //map p to an axis-aligned cube from -1 to 1
        glm::vec3 half_dim(dim.x, dim.y, dim.z);
        half_dim *= 0.5f;
        glm::vec3 ret = (glm::vec3(id.x, id.y, id.z) - half_dim) / half_dim;
        return ret;
    };

    auto RotMat = [](glm::vec3 axis, float angle) -> glm::mat3 {
        // http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
        axis = glm::normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;

        return glm::mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
                         oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
                         oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
    };

    auto sgn = [](float x) -> float {
        return (x < 0.) ? -1. : 1.;
    };

    auto sdJetBox = [](glm::vec3 p, glm::vec3 b) -> float {
        glm::vec3 d = abs(p) - b;
        return std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f) + glm::length(glm::max(d, 0.0f));
    };

    auto sdJetTorus = [](glm::vec3 p, glm::vec2 t) -> float {
        glm::vec2 q = glm::vec2(glm::length(p.xz()) - t.x, p.y);
        return glm::length(q) - t.y;
    };


    auto sdJetCapsule = [&](glm::vec3 p, glm::vec3 a, glm::vec3 b, float r) -> float {
        glm::vec3 pa = p - a, ba = b - a;
        float     h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
        return glm::length(pa - ba * h) - r;
    };

    auto sdJetEllipsoid = [&](glm::vec3 p, glm::vec3 r) -> float {
        return (glm::length(p / r.xyz()) - 1.0f) * r.y;
    };


    auto sdJetConeSection = [&](glm::vec3 p, float h, float r1, float r2) -> float {
        float d1 = -p.z - h;
        float q = p.z - h;
        float si = 0.5f * (r1 - r2) / h;
        float d2 = std::max(std::sqrtf(glm::dot(p.xy(), p.xy()) * (1.0f - si * si)) + q * si - r2, q);
        return glm::length(glm::max(glm::vec2(d1, d2), 0.0f)) + std::min(std::max(d1, d2), 0.f);
    };

    auto fCylinder = [](glm::vec3 p, float r, float height) -> float {
        float d = glm::length(p.xy()) - r;
        d = std::max(d, std::abs(p.z) - height);
        return d;
    };

    auto fSphere = [](glm::vec3 p, float r) -> float {
        return glm::length(p) - r;
    };


    auto sdJetHexPrism = [](glm::vec3 p, glm::vec2 h) -> float {
        glm::vec3 q = glm::abs(p);
        return std::max(q.y - h.y, std::max((q.z * 0.866025f + q.x * 0.5f), q.x) - h.x);
    };

    auto fOpPipe = [](float a, float b, float r) -> float {
        return glm::length(glm::vec2(a, b)) - r;
    };


    auto pModPolar = [](glm::vec2 p, float repetitions) -> glm::vec2 {
        float angle = 2. * glm::pi<float>() / repetitions;
        float a = glm::atan(p.y, p.x) + angle / 2.f;
        float r = glm::length(p);
        float c = std::floor(a / angle);
        a = glm::mod(a, angle) - angle / 2.f;
        p = glm::vec2(glm::cos(a), glm::sin(a)) * r;
        if (std::abs(c) >= (repetitions / 2.)) {
            c = std::abs(c);
        }
        return p;
    };

    auto pModInterval1 = [](float& p, float size, float start, float stop) -> float {
        float halfsize = size * 0.5;
        float c = std::floor((p + halfsize) / size);
        p = glm::mod(p + halfsize, size) - halfsize;
        if (c > stop) {
            p += size * (c - stop);
            c = stop;
        }
        if (c < start) {
            p += size * (c - start);
            c = start;
        }
        return c;
    };

    auto pMirror = [sgn](float& p, float dist) -> float {
        float s = sgn(p);
        p = std::abs(p) - dist;
        return s;
    };


    auto r2 = [](float r) -> glm::mat2 {
        float c = glm::cos(r);
        float s = glm::sin(r);
        return glm::mat2(c, s, -s, c);
    };

    auto pR = [r2](glm::vec2& p, float a) {
        p = p * r2(a);
    };

    auto pR_glm = [pR](glm::vec3& p, float a, int idx, int idy) {
        glm::vec2 temp(p[idx], p[idy]);
        pR(temp, a);
        p[idx] = temp[0];
        p[idy] = temp[1];
    };

    auto fOpUnionRound = [](float a, float b, float r) -> float {
        glm::vec2 u = glm::max(glm::vec2(r - a, r - b), glm::vec2(0));
        return std::max(r, std::min(a, b)) - glm::length(u);
    };

    auto fOpIntersectionRound = [](float a, float b, float r) -> float {
        glm::vec2 u = glm::max(glm::vec2(r + a, r + b), glm::vec2(0));
        return std::min(-r, std::max(a, b)) + glm::length(u);
    };

    auto TranslatePos = [&](glm::vec3 p, float _pitch, float _roll) -> glm::vec3 {
        // limited by euler rotation. I wont get a good plane rotation without quaternions! :-(
        pR_glm(p, _roll - glm::pi<float>(), 0, 1);
        p.z += 5.f;
        pR_glm(p, _pitch, 2, 1);
        p.z -= 5.f;
        return p;
    };

    auto MapEsmPod = [&](glm::vec3 p) -> float {
        float dist = fCylinder(p, 0.15f, 1.0f);
        checkPos = p - glm::vec3(0.f, 0.f, -1.0f);
        pModInterval1(checkPos.z, 2.0f, .0f, 1.0f);
        return std::min(dist, sdJetEllipsoid(checkPos, glm::vec3(0.15f, 0.15f, .5f)));
    };

    auto MapMissile = [&](glm::vec3 p) -> float {
        float d = fCylinder(p, 0.70f, 1.7f);
        if (d < 1.0f) {
            missileDist = std::min(missileDist, fCylinder(p, 0.12f, 1.2f));
            missileDist = std::min(missileDist, sdJetEllipsoid(p - glm::vec3(0.f, 0.f, 1.10f), glm::vec3(0.12f, 0.12f, 1.0f)));

            checkPos = p;
            pR_glm(checkPos, 0.785f, 0, 1);
            checkPos.xy = pModPolar(checkPos.xy, 4.0f);

            missileDist = std::min(missileDist, sdJetHexPrism(checkPos - glm::vec3(0.f, 0.f, .60f), glm::vec2(0.50f, 0.01f)));
            missileDist = std::min(missileDist, sdJetHexPrism(checkPos + glm::vec3(0.f, 0.f, 1.03f), glm::vec2(0.50f, 0.01f)));
            missileDist = std::max(missileDist, -sdJetBox(p + glm::vec3(0.f, 0.f, 3.15f), glm::vec3(3.0f, 3.0f, 2.0f)));
            missileDist = std::max(missileDist, -fCylinder(p + glm::vec3(0.f, 0.f, 2.15f), 0.09f, 1.2f));
        }
        return missileDist;
    };

    auto MapFrontWing = [&](glm::vec3 p, float mirrored) -> float {
        missileDist = 10000.0f;

        checkPos = p;
        pR_glm(checkPos, -0.02f, 0, 1);
        float wing = sdJetBox(checkPos - glm::vec3(4.50f, 0.25f, -4.6f), glm::vec3(3.75f, 0.04f, 2.6f));

        if (wing < 5.f)  //Bounding Box test
        {
            // cutouts
            checkPos = p - glm::vec3(3.0f, 0.3f, -.30f);
            pR_glm(checkPos, -0.5f, 0, 2);
            wing = fOpIntersectionRound(wing, -sdJetBox(checkPos, glm::vec3(6.75f, 1.4f, 2.0f)), 0.1f);

            checkPos = p - glm::vec3(8.0f, 0.3f, -8.80f);
            pR_glm(checkPos, -0.05f, 0, 2);
            wing = fOpIntersectionRound(wing, -sdJetBox(checkPos, glm::vec3(10.75f, 1.4f, 2.0f)), 0.1f);

            checkPos = p - glm::vec3(9.5f, 0.3f, -8.50f);
            wing = fOpIntersectionRound(wing, -sdJetBox(checkPos, glm::vec3(2.0f, 1.4f, 6.75f)), 0.6f);

            // join wing and engine
            wing = std::min(wing, sdJetCapsule(p - glm::vec3(2.20f, 0.3f, -4.2f), glm::vec3(0.f, 0.f, -1.20f), glm::vec3(0.f, 0.f, 0.8f), 0.04f));
            wing = std::min(wing, sdJetCapsule(p - glm::vec3(3.f, 0.23f, -4.2f), glm::vec3(0.f, 0.f, -1.20f), glm::vec3(0.f, 0.f, 0.5f), 0.04f));

            checkPos = p;
            pR_glm(checkPos, -0.03f, 0, 2);
            wing = std::min(wing, sdJetConeSection(checkPos - glm::vec3(0.70f, -0.1f, -4.52f), 5.0f, 0.25f, 0.9f));

            checkPos = p;
            pR_glm(checkPos, 0.75f, 1, 2);
            wing = fOpIntersectionRound(wing, -sdJetBox(checkPos - glm::vec3(3.0f, -.5f, 1.50f), glm::vec3(3.75f, 3.4f, 2.0f)), 0.12f);
            pR_glm(checkPos, -1.95f, 1, 2);
            wing = fOpIntersectionRound(wing, -sdJetBox(checkPos - glm::vec3(2.0f, .70f, 2.20f), glm::vec3(3.75f, 3.4f, 2.0f)), 0.12f);

            checkPos = p - glm::vec3(0.47f, 0.0f, -4.3f);
            pR_glm(checkPos, 1.57f, 1, 2);
            wing = std::min(wing, sdJetTorus(checkPos - glm::vec3(0.0f, -3.f, .0f), glm::vec2(.3f, 0.05f)));

            // flaps
            wing = std::max(wing, -sdJetBox(p - glm::vec3(3.565f, 0.1f, -6.4f), glm::vec3(1.50f, 1.4f, .5f)));
            wing = std::max(wing, -std::max(sdJetBox(p - glm::vec3(5.065f, 0.1f, -8.4f), glm::vec3(0.90f, 1.4f, 2.5f)), -sdJetBox(p - glm::vec3(5.065f, 0.f, -8.4f), glm::vec3(0.89f, 1.4f, 2.49f))));

            checkPos = p - glm::vec3(3.565f, 0.18f, -6.20f + 0.30f);
            pR_glm(checkPos, -0.15f + (0.8f * pitch), 1, 2);
            wing = std::min(wing, sdJetBox(checkPos + glm::vec3(0.0f, 0.0f, 0.30f), glm::vec3(1.46f, 0.007f, 0.3f)));

            // missile holder
            float holder = sdJetBox(p - glm::vec3(3.8f, -0.26f, -4.70f), glm::vec3(0.04f, 0.4f, 0.8f));

            checkPos = p;
            pR_glm(checkPos, 0.85f, 1, 2);
            holder = std::max(holder, -sdJetBox(checkPos - glm::vec3(2.8f, -1.8f, -3.0f), glm::vec3(1.75f, 1.4f, 1.0f)));
            holder = std::max(holder, -sdJetBox(checkPos - glm::vec3(2.8f, -5.8f, -3.0f), glm::vec3(1.75f, 1.4f, 1.0f)));
            holder = fOpUnionRound(holder, sdJetBox(p - glm::vec3(3.8f, -0.23f, -4.70f), glm::vec3(1.0f, 0.03f, 0.5f)), 0.1f);

            // bomb
            bombDist = fCylinder(p - glm::vec3(3.8f, -0.8f, -4.50f), 0.35f, 1.f);
            bombDist = std::min(bombDist, sdJetEllipsoid(p - glm::vec3(3.8f, -0.8f, -3.50f), glm::vec3(0.35f, 0.35f, 1.0f)));
            bombDist = std::min(bombDist, sdJetEllipsoid(p - glm::vec3(3.8f, -0.8f, -5.50f), glm::vec3(0.35f, 0.35f, 1.0f)));

            // missiles
            checkPos = p - glm::vec3(2.9f, -0.45f, -4.50f);

            // check if any missile has been fired. If so, do NOT mod missile position
            float maxMissiles = 0.f;
            if (mirrored > 0.f) {
                maxMissiles = glm::mix(1.0f, 0.f, glm::step(1.f, missilesLaunched.x));
            } else {
                maxMissiles = glm::mix(1.0f, 0.f, glm::step(1.f, missilesLaunched.y));
            }

            pModInterval1(checkPos.x, 1.8f, .0f, maxMissiles);
            holder = std::min(holder, MapMissile(checkPos));

            // ESM Pod
            holder = std::min(holder, MapEsmPod(p - glm::vec3(7.2f, 0.06f, -5.68f)));

            // wheelholder
            wing = std::min(wing, sdJetBox(p - glm::vec3(0.6f, -0.25f, -3.8f), glm::vec3(0.8f, 0.4f, .50f)));

            wing = std::min(bombDist, std::min(wing, holder));
        }

        return wing;
    };

    auto MapRearWing = [&](glm::vec3 p) -> float {
        float wing2 = sdJetBox(p - glm::vec3(2.50f, 0.1f, -8.9f), glm::vec3(1.5f, 0.017f, 1.3f));
        if (wing2 < 0.15f)  //Bounding Box test
        {
            // cutouts
            checkPos = p - glm::vec3(3.0f, 0.0f, -5.9f);
            pR_glm(checkPos, -0.5f, 0, 2);
            wing2 = fOpIntersectionRound(wing2, -sdJetBox(checkPos, glm::vec3(6.75f, 1.4f, 2.0f)), 0.2f);

            checkPos = p - glm::vec3(0.0f, 0.0f, -4.9f);
            pR_glm(checkPos, -0.5f, 0, 2);
            wing2 = fOpIntersectionRound(wing2, -sdJetBox(checkPos, glm::vec3(3.3f, 1.4f, 1.70f)), 0.2f);

            checkPos = p - glm::vec3(3.0f, 0.0f, -11.70f);
            pR_glm(checkPos, -0.05f, 0, 2);
            wing2 = fOpIntersectionRound(wing2, -sdJetBox(checkPos, glm::vec3(6.75f, 1.4f, 2.0f)), 0.1f);

            checkPos = p - glm::vec3(4.30f, 0.0f, -11.80f);
            pR_glm(checkPos, 1.15f, 0, 2);
            wing2 = fOpIntersectionRound(wing2, -sdJetBox(checkPos, glm::vec3(6.75f, 1.4f, 2.0f)), 0.1f);
        }
        return wing2;
    };

    auto MapTailFlap = [&](glm::vec3 p, float mirrored) -> float {
        p.z += 0.3f;
        pR_glm(p, rudderAngle * (-1.f * mirrored), 0, 2);
        p.z -= 0.3f;

        float tailFlap = sdJetBox(p - glm::vec3(0.f, -0.04f, -.42f), glm::vec3(0.025f, .45f, .30f));

        // tailFlap front cutout
        checkPos = p - glm::vec3(0.f, 0.f, 1.15f);
        pR_glm(checkPos, 1.32f, 1, 2);
        tailFlap = std::max(tailFlap, -sdJetBox(checkPos, glm::vec3(.75f, 1.41f, 1.6f)));

        // tailFlap rear cutout
        checkPos = p - glm::vec3(0.f, 0.f, -2.75f);
        pR_glm(checkPos, -0.15f, 1, 2);
        tailFlap = fOpIntersectionRound(tailFlap, -sdJetBox(checkPos, glm::vec3(.75f, 1.4f, 2.0f)), 0.05f);

        checkPos = p - glm::vec3(0.f, 0.f, -.65f);
        tailFlap = std::min(tailFlap, sdJetEllipsoid(checkPos - glm::vec3(0.00f, 0.25f, 0.f), glm::vec3(0.06f, 0.05f, 0.15f)));
        tailFlap = std::min(tailFlap, sdJetEllipsoid(checkPos - glm::vec3(0.00f, 0.10f, 0.f), glm::vec3(0.06f, 0.05f, 0.15f)));

        return tailFlap;
    };

    auto MapTopWing = [&](glm::vec3 p, float mirrored) -> float {
        checkPos = p - glm::vec3(1.15f, 1.04f, -8.5f);
        pR_glm(checkPos, -0.15f, 0, 1);
        float topWing = sdJetBox(checkPos, glm::vec3(0.014f, 0.8f, 1.2f));
        if (topWing < .15f)  //Bounding Box test
        {
            float flapDist = MapTailFlap(checkPos, mirrored);

            checkPos = p - glm::vec3(1.15f, 1.04f, -8.5f);
            pR_glm(checkPos, -0.15f, 0, 1);
            // top border
            topWing = std::min(topWing, sdJetBox(checkPos - glm::vec3(0.f, 0.55f, 0.f), glm::vec3(0.04f, 0.1f, 1.25f)));

            float flapCutout = sdJetBox(checkPos - glm::vec3(0.f, -0.04f, -1.19f), glm::vec3(0.02f, .45f, 1.0f));
            // tailFlap front cutout
            checkPos = p - glm::vec3(1.15f, 2.f, -7.65f);
            pR_glm(checkPos, 1.32f, 1, 2);
            flapCutout = std::max(flapCutout, -sdJetBox(checkPos, glm::vec3(.75f, 1.41f, 1.6f)));

            // make hole for tail flap
            topWing = std::max(topWing, -flapCutout);

            // front cutouts
            checkPos = p - glm::vec3(1.15f, 2.f, -7.f);
            pR_glm(checkPos, 1.02f, 1, 2);
            topWing = fOpIntersectionRound(topWing, -sdJetBox(checkPos, glm::vec3(.75f, 1.41f, 1.6f)), 0.05f);

            // rear cutout
            checkPos = p - glm::vec3(1.15f, 1.f, -11.25f);
            pR_glm(checkPos, -0.15f, 1, 2);
            topWing = fOpIntersectionRound(topWing, -sdJetBox(checkPos, glm::vec3(.75f, 1.4f, 2.0f)), 0.05f);

            // top roll
            topWing = std::min(topWing, sdJetCapsule(p - glm::vec3(1.26f, 1.8f, -8.84f), glm::vec3(0.f, 0.f, -.50f), glm::vec3(0.f, 0.f, 0.3f), 0.06f));

            topWing = std::min(topWing, flapDist);
        }
        return topWing;
    };

    auto MapPlane = [&](glm::vec3 p) -> float {
        float     d = 100000.0f;
        glm::vec3 pOriginal = p;
        // rotate position
        p = TranslatePos(p, pitch, roll);
        float mirrored = 0.f;

        // mirror position at x=0.0. Both sides of the plane are equal.
        mirrored = pMirror(p.x, 0.0f);

        float body = std::min(d, sdJetEllipsoid(p - glm::vec3(0.f, 0.1f, -4.40f), glm::vec3(0.50f, 0.30f, 2.f)));
        body = fOpUnionRound(body, sdJetEllipsoid(p - glm::vec3(0.f, 0.f, .50f), glm::vec3(0.50f, 0.40f, 3.25f)), 1.f);
        body = std::min(body, sdJetConeSection(p - glm::vec3(0.f, 0.f, 3.8f), 0.1f, 0.15f, 0.06f));

        body = std::min(body, sdJetConeSection(p - glm::vec3(0.f, 0.f, 3.8f), 0.7f, 0.07f, 0.01f));

        // window
        winDist = sdJetEllipsoid(p - glm::vec3(0.f, 0.3f, -0.10f), glm::vec3(0.45f, 0.4f, 1.45f));
        winDist = fOpUnionRound(winDist, sdJetEllipsoid(p - glm::vec3(0.f, 0.3f, 0.60f), glm::vec3(0.3f, 0.6f, .75f)), 0.4f);
        winDist = std::max(winDist, -body);
        body = std::min(body, winDist) * 0.8f;
        body = std::min(body, fOpPipe(winDist, sdJetBox(p - glm::vec3(0.f, 0.f, 1.0f), glm::vec3(3.0f, 1.f, .01f)), 0.03f) * 0.7f);
        body = std::min(body, fOpPipe(winDist, sdJetBox(p - glm::vec3(0.f, 0.f, 0.0f), glm::vec3(3.0f, 1.f, .01f)), 0.03f) * 0.7f);

        // front (nose)
        body = std::max(body, -std::max(fCylinder(p - glm::vec3(0.f, 0.f, 2.5f), .46f, 0.04f), -fCylinder(p - glm::vec3(0.f, 0.f, 2.5f), .35f, 0.1f)));
        checkPos = p - glm::vec3(0.f, 0.f, 2.5f);
        pR_glm(checkPos, 1.57f, 1, 2);
        body = fOpIntersectionRound(body, -sdJetTorus(checkPos + glm::vec3(0.f, 0.80f, 0.f), glm::vec2(.6f, 0.05f)), 0.015f);
        body = fOpIntersectionRound(body, -sdJetTorus(checkPos + glm::vec3(0.f, 2.30f, 0.f), glm::vec2(.62f, 0.06f)), 0.015f);

        // wings
        frontWingDist = MapFrontWing(p, mirrored);
        d = std::min(d, frontWingDist);
        rearWingDist = MapRearWing(p);
        d = std::min(d, rearWingDist);
        topWingDist = MapTopWing(p, mirrored);
        d = std::min(d, topWingDist);

        // bottom
        checkPos = p - glm::vec3(0.f, -0.6f, -5.0f);
        pR_glm(checkPos, 0.07f, 1, 2);
        d = fOpUnionRound(d, sdJetBox(checkPos, glm::vec3(0.5f, 0.2f, 3.1f)), 0.40f);

        float holder = sdJetBox(p - glm::vec3(0.0f, -1.1f, -4.30f), glm::vec3(0.08f, 0.4f, 0.8f));
        checkPos = p;
        pR_glm(checkPos, 0.85f, 1, 2);
        holder = std::max(holder, -sdJetBox(checkPos - glm::vec3(0.0f, -5.64f, -2.8f), glm::vec3(1.75f, 1.4f, 1.0f)));
        d = fOpUnionRound(d, holder, 0.25f);

        // large bomb
        bombDist2 = fCylinder(p - glm::vec3(0.0f, -1.6f, -4.0f), 0.45f, 1.0f);
        bombDist2 = std::min(bombDist2, sdJetEllipsoid(p - glm::vec3(0.0f, -1.6f, -3.20f), glm::vec3(0.45f, 0.45f, 2.f)));
        bombDist2 = std::min(bombDist2, sdJetEllipsoid(p - glm::vec3(0.0f, -1.6f, -4.80f), glm::vec3(0.45f, 0.45f, 2.f)));

        d = std::min(d, bombDist2);

        d = std::min(d, sdJetEllipsoid(p - glm::vec3(1.05f, 0.13f, -8.4f), glm::vec3(0.11f, 0.18f, 1.0f)));

        checkPos = p - glm::vec3(0.f, 0.2f, -5.0f);
        d = fOpUnionRound(d, fOpIntersectionRound(sdJetBox(checkPos, glm::vec3(1.2f, 0.14f, 3.7f)), -sdJetBox(checkPos, glm::vec3(1.f, 1.14f, 4.7f)), 0.2f), 0.25f);

        d = fOpUnionRound(d, sdJetEllipsoid(p - glm::vec3(0.f, 0.f, -4.f), glm::vec3(1.21f, 0.5f, 2.50f)), 0.75f);

        // engine cutout
        blackDist = std::max(d, fCylinder(p - glm::vec3(.8f, -0.15f, 0.f), 0.5f, 2.4f));
        d = std::max(d, -fCylinder(p - glm::vec3(.8f, -0.15f, 0.f), 0.45f, 2.4f));

        // engine
        d = std::max(d, -sdJetBox(p - glm::vec3(0.f, 0.f, -9.5f), glm::vec3(1.5f, 0.4f, 0.7f)));

        engineDist = fCylinder(p - glm::vec3(0.40f, -0.1f, -8.7f), .42f, 0.2f);
        checkPos = p - glm::vec3(0.4f, -0.1f, -8.3f);
        pR_glm(checkPos, 1.57f, 1, 2);
        engineDist = std::min(engineDist, sdJetTorus(checkPos, glm::vec2(.25f, 0.25f)));
        engineDist = std::min(engineDist, sdJetConeSection(p - glm::vec3(0.40f, -0.1f, -9.2f), 0.3f, .22f, .36f));

        checkPos = p - glm::vec3(0.f, 0.f, -9.24f);
        checkPos.xy -= glm::vec2(0.4f, -0.1f);
        checkPos.xy = pModPolar(checkPos.xy(), 22.0f);

        float engineCone = fOpPipe(engineDist, sdJetBox(checkPos, glm::vec3(.6f, 0.001f, 0.26f)), 0.015f);
        engineDist = std::min(engineDist, engineCone);

        d = std::min(d, engineDist);

        d = std::min(d, winDist);
        d = std::min(d, body);

        d = std::min(d, sdJetBox(p - glm::vec3(1.1f, 0.f, -6.90f), glm::vec3(.33f, .12f, .17f)));
        checkPos = p - glm::vec3(0.65f, 0.55f, -1.4f);
        pR_glm(checkPos, -0.35f, 1, 2);
        d = std::min(d, sdJetBox(checkPos, glm::vec3(0.2f, 0.1f, 0.45f)));

        return d;
    };


    glm::vec3 p = mapToCube(id);
    p.z += 0.8;

    p = p * RotMat(glm::vec3(0.f, 1.f, 0.f), glm::pi<float>());
    const float scale = 0.12;
    p *= (1.0 / scale);

    return MapPlane(p) * 0.9 * scale;
}