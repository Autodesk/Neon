#include "Neon/skeleton/internal/dependencyTools/DependencyAnalyser.h"


namespace Neon::skeleton::internal {


DependencyAnalyser::
    DependencyAnalyser(Neon::set::dataDependency::MultiXpuDataUid uid,
                       Neon::set::dataDependency::MultiXpuDataIdx idx)
{
    mUid = uid;
    mIdx = idx;
}

auto DependencyAnalyser::
    update(Neon::set::container::GraphInfo::NodeUid newKernel,
           Neon::set::dataDependency::AccessType    newOp)
        -> std::vector<DataDependency>
{
    switch (newOp) {
        case Neon::set::dataDependency::AccessType::READ: {
            if (mParsedW.size() == 0) {
                // We are parsing a READ with no previous WRITE
                // STEPS:
                // a. Register the current kernel in the READ state machine
                // b. Returns no dependencies

                // Executing step a.
                mParsedR.push_back(newKernel);
                // Executing step b.
                return std::vector<DataDependency>(0);
            }
            if (mParsedW.size() == 1) {
                std::vector<DataDependency> output;
                // Parsing a READ after a WRITE
                // STEPS:
                // a. Return a RaW between the new READing kernel and the old WRITing kernel
                // b. Register the current kernel in the READ state machine
                //------


                {  // Executing a.
                    auto t0W = mParsedW[0];
                    auto t1R = newKernel;
                    if (t0W != t1R) {
                        DataDependency d(t1R, Neon::set::dataDependency::DataDependencyType::RAW, mUid, t0W);
                        output.push_back(d);
                    }
                }

                {  // Executing b.
                    mParsedR.push_back(newKernel);
                }
                return output;

            } else {
                NEON_THROW_UNSUPPORTED_OPERATION("A Violation of the state machine integrity was detected");
            }
            break;
        }
        case Neon::set::dataDependency::AccessType::WRITE: {
            if (mParsedR.empty() && mParsedW.empty()) {
                // Parsing a WRITE as the first operation in the Container sequence.
                //
                // STEPS:
                // a. Record the Write operation in the state machine
                // b. Return no dependency
                //------

                {  // Executing a.
                    mParsedW.push_back(newKernel);
                }

                {  // Executing b.
                    return std::vector<DataDependency>(0);
                }
            }

            if (!mParsedR.empty() && mParsedW.empty()) {
                std::vector<DataDependency> output;
                // Parsing a WRITE after some READs
                //

                // STEPS:
                // a. creation the vector of WAR dependencies to be returned
                // b. clear state machine state, cleaning previous read token and storing the new write token

                {  // Executing a.
                    for (const auto token_t0_READ : mParsedR) {
                        const auto token_t1_WRITE = newKernel;
                        if (token_t1_WRITE != token_t0_READ) {
                            DataDependency d(token_t1_WRITE, Neon::set::dataDependency::DataDependencyType::WAR, mUid, token_t0_READ);
                            output.push_back(d);
                        }
                    }
                }

                {  // Executing b.
                    mParsedR.clear();
                    mParsedW = std::vector<Neon::set::container::GraphInfo::NodeUid>(1, newKernel);
                }

                return output;
            }

            if (!mParsedW.empty()) {
                std::vector<DataDependency> output;
                // STEPS:
                // a. flag a WaW dependency
                // b. flag possible WaR dependencies
                // b. clear state machine state, cleaning previous read token and storing the new write token

                //NEON_WARNING("Skeleton: WaW dependency detected.");
                {  // Executing Step a.
                    auto token_t0_WRITE = mParsedW[0];
                    auto token_t1_WRITE = newKernel;
                    if (token_t0_WRITE != token_t1_WRITE) {
                        DataDependency d(token_t1_WRITE, Neon::set::dataDependency::DataDependencyType::WAW, mUid, token_t0_WRITE);
                        output.push_back(d);
                    }
                }
                {  // Executing Step b.
                    for (const auto token_t0_READ : mParsedR) {
                        const auto token_t1_WRITE = newKernel;
                        if (token_t1_WRITE != token_t0_READ) {
                            DataDependency d(token_t1_WRITE, Neon::set::dataDependency::DataDependencyType::WAR, mUid, token_t0_READ);
                            output.push_back(d);
                        }
                    }
                }
                {  // Executing c.

                    mParsedR.clear();
                    mParsedW = std::vector<Neon::set::container::GraphInfo::NodeUid>(1, newKernel);
                }
                return output;
            }
            break;
        }
        case Neon::set::dataDependency::AccessType::NONE: {
            // Error
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}


}  // namespace Neon::skeleton::internal