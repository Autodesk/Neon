#include <mpi.h>
#include <nccl.h>
#include <iostream>
#include "Neon/Neon.h"
#include "Neon/domain/details/ncclGrid/ncclGrid.h"
#include "backend.h"


template <typename T>
auto newConatiner(Neon::domain::details::ncclGrid::ncclField<T, 0> fieldA,
                  Neon::domain::details::ncclGrid::ncclField<T, 0> fieldB) -> Neon::set::Container
{
    const auto& grid = fieldA.getGrid();
    using Field = Neon::domain::details::ncclGrid::ncclField<T, 0>;
    return grid.newContainer(
        "HU-test",
        [&](Neon::set::Loader& loader) {
            using Ngh3DIdx = Neon::int8_3d;
            const auto d_fa = loader.load(fieldA);
            auto       d_fb = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& pIdx) mutable {
                auto                                    gIdx = d_fa.getGlobalIndex(pIdx);
                Neon::index_3d                          center(d_fa(pIdx, 0), d_fa(pIdx, 1), d_fa(pIdx, 2));
                Neon::index_3d                          pass(0, 0, 0);
                constexpr std::array<const Ngh3DIdx, 6> stencil{
                    Ngh3DIdx(1, 0, 0),
                    Ngh3DIdx(-1, 0, 0),
                    Ngh3DIdx(0, 1, 0),
                    Ngh3DIdx(0, -1, 0),
                    Ngh3DIdx(0, 0, 1),
                    Ngh3DIdx(0, 0, -1)};

                for (auto const& direction : stencil) {
                    Neon::index_3d                          passPerDirection(0, 0, 0);
                    Neon::index_3d ngh(0, 0, 0);
                    auto           expected = center + direction.newType<int32_t>();
                    for (int i = 0; i < 3; i++) {
                        typename Field::NghData nghData = d_fa.getNghData(pIdx, direction, i);
                        if (nghData.isValid()) {
                            ngh.getVectorView()[i] = nghData.getData();
                            if (ngh.getVectorView()[i] != expected.getVectorView()[i]) {
                                passPerDirection.getVectorView()[i] = 1;
                                pass.getVectorView()[i] = 1;
                            }
                        }
                    }
                    if (passPerDirection != Neon::index_3d(0,0,0)) {
                        printf("Error! at %d %d %d Expected %d %d %d Found %d %d %d\n", gIdx.x, gIdx.y, gIdx.z, expected.x, expected.y, expected.z, ngh.x, ngh.y, ngh.z);
                    }else {
                        //printf("Pass! at %d %d %d Expected %d %d %d Found %d %d %d\n", gIdx.x, gIdx.y, gIdx.z, expected.x, expected.y, expected.z, ngh.x, ngh.y, ngh.z);
                    }
                }
                for (int i = 0; i < 3; i++) {
                    d_fb(pIdx, i) = pass.getVectorView()[i];
                }
            };
        });
}


int main(int /*argc*/, char** /*argv*/)
{
    if (false) {
        Neon::Backend                             bk(Neon::Runtime::stream);
        Neon::domain::details::ncclGrid::ncclGrid grid(
            bk,
            Neon::int32_3d(100, 100, 100),
            [&](Neon::index_3d const& /*idx*/) -> bool { return true; },
            Neon::domain::Stencil::s7_Laplace_t());

        auto field = grid.template newField<int>("test", 1, 0);
        field.forEachActiveCell([](const Neon::index_3d& idx, auto& values) {
            *values[0] = idx.x + idx.y + idx.z;
        });

        field.updateDeviceData(0);
        field.ioToVtk("test", "test");
        auto hu = field.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                      Neon::set::TransferMode::get,
                                      Neon::Execution::device);

        hu.run(0);
        bk.sync(0);
    } else {
        Neon::Backend                             bk(Neon::Runtime::stream);
        Neon::domain::details::ncclGrid::ncclGrid grid(
            bk,
            Neon::int32_3d(10, 10, 10),
            [&](Neon::index_3d const& /*idx*/) -> bool { return true; },
            Neon::domain::Stencil::s7_Laplace_t());

        auto fA = grid.template newField<int>("test", 3, 0);
        auto fB = grid.template newField<int>("test", 3, 0);

        fA.forEachActiveCell([](const Neon::index_3d& idx, auto& values) {
            *values[0] = idx.x;
            *values[1] = idx.y;
            *values[2] = idx.z;
        });

        fA.updateDeviceData(0);
        fA.ioToVtk("test", "test");
        bk.sync(0);

        auto hu = fA.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                   Neon::set::TransferMode::get,
                                   Neon::Execution::device);

        hu.run(0);
        bk.sync(0);

        auto test = newConatiner(fA, fB);
        test.run(0);
        bk.sync(0);
    }
}

#if 0
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the number of available GPUs
    int num_devices;
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");

    // Assign a GPU to each MPI rank
    int local_rank = world_rank % num_devices;
    checkCuda(cudaSetDevice(local_rank), "cudaSetDevice");

    std::cout << "MPI Rank " << world_rank
              << " using GPU " << local_rank
              << " of " << num_devices << std::endl;

    // NCCL initialization
    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm;

    if (world_rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }

    // Broadcast NCCL ID from rank 0 to all other ranks
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    checkNccl(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank), "ncclCommInitRank");

    std::cout << "NCCL initialized for rank " << world_rank << std::endl;

    // NCCL communication (example: simple all-reduce)
    float send_data = world_rank + 1.0f;
    float recv_data = 0.0f;
    checkNccl(ncclAllReduce(&send_data, &recv_data, 1, ncclFloat, ncclSum, nccl_comm, 0), "ncclAllReduce");

    cudaDeviceSynchronize();
    std::cout << "Rank " << world_rank << " received sum: " << recv_data << std::endl;

    // Finalize NCCL and MPI
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();

    return 0;
}
#endif