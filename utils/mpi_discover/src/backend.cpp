#include "backend.h"
#include <mpi.h>
#include <nccl.h>

#include "Neon/set/DevSet.h"
#include "Neon/Neon.h"

namespace Neon {
namespace distributed {

void checkNccl(ncclResult_t result, const char* func)
{
    if (result != ncclSuccess) {
        std::cerr << "NCCL error in " << func << ": " << ncclGetErrorString(result) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

void checkCuda(cudaError_t result, const char* func)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}


struct Backend::Data
{
    int           worldRank;
    int           worldSize;
    Neon::Backend backend;
    ncclUniqueId  nccl_id;
    ncclComm_t    nccl_comm;
    int           numLocalDevices;
    int           sizeLocalRanks;
    int           localRank;
    int           localSize;

    Data()
    {
        worldRank = 0;
        worldSize = 0;
        numLocalDevices = 0;
        sizeLocalRanks = 0;
        localRank = 0;
        localSize = 0;
    }
};



auto initMPI(Backend::Data& data) -> void
{
    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &data.worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &data.worldSize);

    {  // loca rank info
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &data.localRank);
        MPI_Comm_size(local_comm, &data.localSize);
        MPI_Comm_free(&local_comm);
        MPI_Barrier(MPI_COMM_WORLD);

    }
    {  // Get the number of available GPUs
        checkCuda(cudaGetDeviceCount(&data.numLocalDevices), "cudaGetDeviceCount");
        std::cout << "MPI Rank " << data.worldRank
                  << " using GPU " << data.localRank
                  << " of " << data.numLocalDevices << std::endl;
    }


    if (data.sizeLocalRanks > data.numLocalDevices) {
        std::cout <<"error..." << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        Neon::NeonException ex("distributed::Backend");
        ex << "Unsupported configuration. At the moment we require the number of devices (";
        ex << data.numLocalDevices;
        ex << ") to be the same as the MPI rank per node";
    }else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << "MPI Rank " << data.worldRank
              << "MPI init completed" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

}

auto initNCCL(Backend::Data& data) -> void
{
    if (data.worldRank == 0) {
        std::cout << "MPI rank 0 - ncclGetUniqueId " << std::endl;
        ncclGetUniqueId(&(data.nccl_id));
    }

    // Broadcast NCCL ID from rank 0 to all other ranks
    MPI_Bcast(&(data.nccl_id), sizeof(data.nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize NCCL communicator
    checkNccl(ncclCommInitRank(&(data.nccl_comm),
                               data.worldSize,
                               data.nccl_id,
                               data.worldRank),
              "initNCCL");
    std::cout << "NCCL initialized for rank " << data.worldRank << std::endl;
}

Backend::Backend()
{
    Neon::init();
    mData = std::make_shared<Data>();
    initMPI(*mData);
    std::vector<int> myDeviceList;
    myDeviceList.push_back(mData->localRank);
    mData->backend = Neon::Backend(myDeviceList, Neon::Runtime::stream);
    auto const& devSet = mData->backend.devSet();
    devSet.setActiveDevContext(0);
    initNCCL(*mData);

    //
    // // NCCL communication (example: simple all-reduce)
    // float send_data = world_rank + 1.0f;
    // float recv_data = 0.0f;
    // checkNccl(ncclAllReduce(&send_data, &recv_data, 1, ncclFloat, ncclSum, nccl_comm, 0), "ncclAllReduce");
    //
    // cudaDeviceSynchronize();
    // std::cout << "Rank " << world_rank << " received sum: " << recv_data << std::endl;

    // Finalize NCCL and MPI
    // ncclCommDestroy(nccl_comm);
    // MPI_Finalize();
}
Backend::~Backend()
{
    ncclCommDestroy(mData->nccl_comm);
    MPI_Finalize();
}

}  // namespace distributed
}  // namespace Neon