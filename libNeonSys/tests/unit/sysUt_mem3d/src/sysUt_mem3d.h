#pragma once
#include "gtest/gtest.h"

#include "Neon/core/types/vec.h"

#include "Neon/sys/global/CpuSysGlobal.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include "Neon/sys/memory/mem3d.h"

#include <cstring>
#include <iostream>

namespace global {

std::vector<int> Ls{10, 15, 20, 25, 30, 35};
std::vector<int> As{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

auto checkAlignment = [](auto& Mem3d, int A) {
    auto   Abyte = Neon::sys::MemAlignment::exp2byte(A);
    size_t reminder = 0;
    size_t addr = size_t(Mem3d.mem());

    reminder = addr % Abyte;

    ASSERT_TRUE(reminder == 0) << "Address: " << (void*)addr << " reminder " << reminder << "\n";
};

auto checkAllocation = [](auto& Mem3d) {
    Mem3d.memset(0x0);
};

auto haloTest = [](int L, int A, Neon::DeviceType devEt, Neon::Allocator allocEt) {
    Neon::index_3d          dim(L);
    Neon::index_3d          halo(1, 1, 1);
    Neon::sys::Mem3d_t<int> mem(1, devEt, Neon::sys::DeviceID(0), allocEt, dim, halo,
                                Neon::memLayout_et::structOfArrays,
                                Neon::sys::MemAlignment(Neon::sys::memAlignment_et::user, A),
                                Neon::memLayout_et::OFF);
    checkAllocation(mem);
    checkAlignment(mem, A);
};
auto noHaloTest = [](int L, int A, Neon::DeviceType devEt, Neon::Allocator allocEt) {
    Neon::index_3d          dim(L);
    Neon::index_3d          halo(0, 0, 0);
    Neon::sys::Mem3d_t<int> mem(1, devEt, Neon::sys::DeviceID(0), allocEt, dim, halo,
                                Neon::memLayout_et::structOfArrays,
                                Neon::sys::MemAlignment(Neon::sys::memAlignment_et::user, A),
                                Neon::memLayout_et::OFF);
    checkAllocation(mem);
    checkAlignment(mem, A);
};

auto copyTest = [](int L, int val, Neon::DeviceType devEt, Neon::Allocator allocEt) {
    Neon::index_3d dim(L);
    Neon::index_3d halo(0, 0, 0);

    Neon::sys::Mem3d_t<int> mem_A(1, devEt, Neon::sys::DeviceID(0), allocEt, dim, halo);
    mem_A.memset(0);

    Neon::sys::Mem3d_t<int> mem_B(1, devEt, Neon::sys::DeviceID(0), allocEt, dim, halo);

    for (int k = 0; k < dim.z; ++k) {
        for (int j = 0; j < dim.y; ++j) {
            for (int i = 0; i < dim.x; ++i) {
                mem_B.elRef(Neon::index_3d(i, j, k)) = val;
            }
        }
    }

    Neon::sys::GpuDevice   gpuDev(0);
    Neon::sys::GpuStream   gpuStream = gpuDev.tools.stream();
    //ASSERT_NO_THROW(mem_A.copyFrom(mem_B));
    ASSERT_NO_THROW(mem_A.fastCopyFrom<Neon::run_et::sync>(gpuStream, mem_B));

    for (int k = 0; k < dim.z; ++k) {
        for (int j = 0; j < dim.y; ++j) {
            for (int i = 0; i < dim.x; ++i) {
                EXPECT_EQ(mem_A.elRef(Neon::index_3d(i, j, k)), val);
            }
        }
    }
};
}  // namespace global


TEST(mem3d, CPU)
{
    for (auto&& L : global::Ls) {
        for (auto&& A : global::As) {
            ASSERT_NO_THROW(global::haloTest(L, A, Neon::DeviceType::CPU, Neon::Allocator::MALLOC));
            ASSERT_NO_THROW(global::haloTest(L, A, Neon::DeviceType::CPU, Neon::Allocator::CUDA_MEM_HOST));

            ASSERT_NO_THROW(global::noHaloTest(L, A, Neon::DeviceType::CPU, Neon::Allocator::MALLOC));
            ASSERT_NO_THROW(global::noHaloTest(L, A, Neon::DeviceType::CPU, Neon::Allocator::CUDA_MEM_HOST));

            ASSERT_NO_THROW(global::copyTest(L, 10, Neon::DeviceType::CPU, Neon::Allocator::MALLOC));
            ASSERT_NO_THROW(global::copyTest(L, 10, Neon::DeviceType::CPU, Neon::Allocator::CUDA_MEM_HOST));
        }
    }
}


TEST(mem3d, GPU)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        for (auto&& L : global::Ls) {
            for (auto&& A : global::As) {
                ASSERT_NO_THROW(global::haloTest(L, A, Neon::DeviceType::CUDA, Neon::Allocator::CUDA_MEM_DEVICE));
                ASSERT_NO_THROW(global::noHaloTest(L, A, Neon::DeviceType::CUDA, Neon::Allocator::CUDA_MEM_DEVICE));
            }
        }
    }
}


TEST(mem3d, exportVTI)
{
    int            L = 10;
    double         val = 100.0;
    Neon::index_3d dim(L);
    Neon::index_3d halo(1, 1, 1);

    Neon::sys::Mem3d_t<double> mem3d(1, Neon::DeviceType::CPU,
                                     Neon::sys::DeviceID(0), Neon::Allocator::MALLOC, dim, halo);
    mem3d.memset(0);

    for (int k = 0; k < dim.z; ++k) {
        for (int j = 0; j < dim.y; ++j) {
            for (int i = 0; i < dim.x; ++i) {
                if (double(i * i + j * j + k * k) < double(L * L) * 0.25) {
                    mem3d.elRef(Neon::index_3d(i, j, k)) = val;
                }
            }
        }
    }

    mem3d.exportVti<Neon::vti_e::VOXEL>(true, "Testmem3d_withHalo");
    mem3d.exportVti<Neon::vti_e::VOXEL>(false, "Testmem3d_nohalo");
}