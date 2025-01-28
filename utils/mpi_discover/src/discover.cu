#include "../../../libNeonSkeleton/tests/perf/SkeletonSyntheticBenchmarks/src/CLiCorrectness.h"
#include "Neon/domain/details/bGridDisg/BlockView.h"
#include "Neon/domain/details/ncclGrid/ncclGrid.h"
#include "Neon/set/Containter.h"

/**
 * A simple tutorial demonstrating the use of staggered grid in Neon.
 */
using Field = Neon::domain::details::ncclGrid::ncclField<int>;
auto laplaceTemplate(const Field& filedA,
                     Field&       fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "stencilFun",
        [&](Neon::set::Loader& loader) {
            const auto aP = loader.load(filedA, Neon::Pattern::STENCIL);
            auto       bP = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& gIdx) mutable {
                auto          globalIdx = aP.getGlobalIndex(gIdx);
                Neon::int8_3d offseets[6];
                offseets[0] = {1, 0, 0};
                offseets[1] = {-1, 0, 0};
                offseets[2] = {0, 1, 0};
                offseets[3] = {0, -1, 0};
                offseets[4] = {0, 0, 1};
                offseets[5] = {0, 0, -1};

                for (int i = 0; i < 6; i++) {

                    auto const&          offset = offseets[i];
                    const Neon::index_3d expected_vals = globalIdx + offset.newType<int>();
                    auto                 nghDataX = aP.getNghData(gIdx, offset, 0, 0);
                    auto                 nghDataY = aP.getNghData(gIdx, offset, 1, 0);
                    auto                 nghDataZ = aP.getNghData(gIdx, offset, 2, 0);

                    if (nghDataX.isValid()) {
                        Neon::index_3d readFromNgh(nghDataX.getData(),
                                                   nghDataY.getData(),
                                                   nghDataZ.getData());


                        readFromNgh.x == expected_vals.x ? bP(gIdx, 0) = 1 : bP(gIdx, 0) = -1;
                        readFromNgh.y == expected_vals.y ? bP(gIdx, 1) = 1 : bP(gIdx, 1) = -1;
                        readFromNgh.z == expected_vals.z ? bP(gIdx, 2) = 1 : bP(gIdx, 2) = -1;
                    } else {
                        bP(gIdx, 0) = 0;
                        bP(gIdx, 1) = 0;
                        bP(gIdx, 2) = 0;
                    }
                }
            };
        });
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