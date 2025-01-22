#include <mpi.h>
#include <iostream>
#include "Neon/Neon.h"
#include <nccl.h>
#include "backend.h"
/**
 * A simple tutorial demonstrating the use of staggered grid in Neon.
 */


int main(int /*argc*/, char** /*argv*/)
{
    Neon::distributed::Backend bk;
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