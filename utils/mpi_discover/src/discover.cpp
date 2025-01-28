#include <mpi.h>
#include <nccl.h>
#include <iostream>
#include "Neon/Neon.h"
#include "Neon/domain/details/ncclGrid/ncclGrid.h"
#include "backend.h"
/**
 * A simple tutorial demonstrating the use of staggered grid in Neon.
 */


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
    }else {
        Neon::Backend                             bk(Neon::Runtime::stream);
        Neon::domain::details::ncclGrid::ncclGrid grid(
            bk,
            Neon::int32_3d(100, 100, 100),
            [&](Neon::index_3d const& /*idx*/) -> bool { return true; },
            Neon::domain::Stencil::s7_Laplace_t());

        auto fA = grid.template newField<int>("test", 3, 0);
        auto fB = grid.template newField<int>("test", 3, 0);

        fA.forEachActiveCell([](const Neon::index_3d& idx, auto& values) {
            *values[0] = idx.x ;
            *values[1] = idx.y ;
            *values[2] = idx.z;
        });

        fA.updateDeviceData(0);
        field.ioToVtk("test", "test");
        bk.sync(0);

        auto hu = fA.newHaloUpdate(Neon::set::StencilSemantic::standard,
            Neon::set::TransferMode::get,
            Neon::Execution::device);

        hu.run(0);
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