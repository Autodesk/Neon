#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/internal/MultiXpuGraph.h"
// #include "Neon/skeleton/internal/StreamScheduler.h"
#ifdef NEON_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
namespace Neon::skeleton {

struct Skeleton
{
   public:
    /**
     * Default empty constructor
     */
    Skeleton() = default;

    /**
     * Constructor that sets the target backend
     * @param bk
     */
    Skeleton(const Neon::Backend& bk);

    /**
     * method to set a backend
     * @param bk
     */
    void setBackend(const Neon::Backend& bk);

    void sequence(const std::vector<Neon::set::Container>& operations,
                  std::string                              name,
                  Options                                  options = Options())
    {
        if (!m_inited) {
            NeonException exp("");
            exp << "A backend was not set";
            NEON_THROW(exp);
        }
        mOptions = options;
        mMultiGraph.init(mBackend, operations, name, options);
        mMultiGraph.ioToDot("DB_multiGpuGraph", "graphname");
        // mStreamScheduler.init(mBackend, mMultiGraph);
        // m_streamScheduler.io2Dot("DB_streamScheduler", "graphname");
    }


    void ioToDot(std::string fname,
                 std::string graphname = "",
                 bool        debug = false)
    {
        mMultiGraph.ioToDot(fname, graphname, debug);
    }

    void run()
    {
#ifdef NEON_USE_NVTX
        nvtxRangePush("Skeleton");
#endif
        mMultiGraph.execute(mOptions);
#ifdef NEON_USE_NVTX
        nvtxRangePop();
#endif
    }

   private:
    Neon::Backend                           mBackend;
    Options                                 mOptions;
    Neon::skeleton::internal::MultiXpuGraph mMultiGraph;
    //    Neon::skeleton::internal::StreamScheduler mStreamScheduler;

    bool m_inited = {false};
};

}  // namespace Neon::skeleton
