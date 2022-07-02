#pragma once
#include "Neon/Report.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Executor.h"
#include "Neon/skeleton/Occ.h"

namespace Neon::skeleton {


struct Options
{
   public:
    /**
     * Constructor that defines options for the skeleton
     * @param occ
     * @param transferMode
     */
    explicit Options(Occ occ, Neon::set::TransferMode transferMode = Neon::set::TransferMode::get);
    explicit Options() = default;

    void reportStore(Neon::Report& report);

    auto occ() const -> Occ;
    auto transferMode() const -> Neon::set::TransferMode;
    auto executor()const -> Neon::skeleton::Executor;

   private:
    Neon::set::TransferMode  mTransferMode{Neon::set::TransferMode::get};
    Neon::skeleton::Occ      mOcc = Occ::none;
    Neon::skeleton::Executor mExecutor = Neon::skeleton::Executor::ompAtNodeLevel;
};

}  // namespace Neon::skeleton