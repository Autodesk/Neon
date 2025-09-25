#include "backend.h"
#include <mpi.h>
#include <nccl.h>

#include "Neon/set/DevSet.h"
#include "Neon/Neon.h"

// namespace Neon {
// Backend::Backend()
// {
//     Neon::init();
//     mData = std::make_shared<Data>();
//     initMPI(*mData);
//     std::vector<int> myDeviceList;
//     myDeviceList.push_back(mData->localRank);
//     mData->backend = Neon::Backend(myDeviceList, Neon::Runtime::stream);
//     auto const& devSet = mData->backend.devSet();
//     devSet.setActiveDevContext(0);
//     initNCCL(*mData);
//
//     //
//     // // NCCL communication (example: simple all-reduce)
//     // float send_data = world_rank + 1.0f;
//     // float recv_data = 0.0f;
//     // checkNccl(ncclAllReduce(&send_data, &recv_data, 1, ncclFloat, ncclSum, nccl_comm, 0), "ncclAllReduce");
//     //
//     // cudaDeviceSynchronize();
//     // std::cout << "Rank " << world_rank << " received sum: " << recv_data << std::endl;
//
//     // Finalize NCCL and MPI
//     // ncclCommDestroy(nccl_comm);
//     // MPI_Finalize();
// }
// Backend::~Backend()
// {
//     ncclCommDestroy(mData->nccl_comm);
//     MPI_Finalize();
// }
//
// }  // namespace distributed
// }  // namespace Neon