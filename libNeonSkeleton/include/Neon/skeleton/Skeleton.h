#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/internal/MultiGpuGraph.h"
#include "Neon/skeleton/internal/StreamScheduler.h"

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
        // m_multiGraph.io2Dot("DB_multiGpuGraph", "graphname");
        mStreamScheduler.init(mBackend, mMultiGraph);
        // m_streamScheduler.io2Dot("DB_streamScheduler", "graphname");
    }


    void ioToDot(std::string fname, std::string graphname = "")
    {
        // m_multiGraph.io2Dot(fname + ".multiGpu.dot", graphname);
        mMultiGraph.io2DotOriginalApp(fname + ".appGraph.dot", graphname);
        mStreamScheduler.io2Dot(fname + ".scheduler.dot", graphname);
        mStreamScheduler.io2DotOrder(fname + ".order.dot", graphname);
    }

    void run()
    {
        mStreamScheduler.run(mOptions);
    }

   private:
    Neon::Backend                             mBackend;
    Options                                   mOptions;
    Neon::skeleton::internal::MultiGpuGraph   mMultiGraph;
    Neon::skeleton::internal::StreamScheduler mStreamScheduler;

    bool m_inited = {false};
};

}  // namespace Neon::skeleton
