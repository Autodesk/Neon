#pragma once

#include "type_traits"

#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/StencilSemantic.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/dependency/AccessType.h"
#include "Neon/set/dependency/ComputeType.h"
#include "Neon/set/dependency/Token.h"

namespace Neon::set {
namespace internal {

struct LoadingMode_e
{
    enum e
    {
        PARSE_AND_EXTRACT_LAMBDA,
        EXTRACT_LAMBDA
    };
    LoadingMode_e() = delete;
    LoadingMode_e(const LoadingMode_e&) = delete;
};

}  // namespace internal


/// https://blog.codeisc.com/2018/01/09/cpp-comma-operator-nice-usages.html

/**
 * Loader that is used to "load" fields into a kernel lambda function
 */
struct Loader
{
    friend Neon::set::DevSet;
    friend Neon::set::internal::ContainerAPI;

   private:
    Neon::set::internal::ContainerAPI& m_container;

    Neon::Execution                       mExecution;
    Neon::SetIdx                          m_setIdx;
    Neon::DataView                        m_dataView;
    Neon::set::internal::LoadingMode_e::e m_loadingMode;

   public:
    Loader(Neon::set::internal::ContainerAPI&    container,
           Neon::Execution                       execution,
           Neon::SetIdx                          setIdx,
           Neon::DataView                        dataView,
           Neon::set::internal::LoadingMode_e::e loadingMode)
        : m_container(container),
          mExecution(execution),
          m_setIdx(setIdx),
          m_dataView(dataView),
          m_loadingMode(loadingMode)
    {
    }

    auto getExecution() const -> Neon::Execution;
    auto getSetIdx() const -> Neon::SetIdx;
    auto getDataView() const -> Neon::DataView;

   public:
    template <typename Field_ta>
    auto load(Field_ta&       field,
              Neon::Compute   computeE = Neon::Compute::MAP,
              StencilSemantic stencilSemantic = StencilSemantic::standard)
        -> std::enable_if_t<!std::is_const_v<Field_ta>, typename Field_ta::Partition&>;

    auto computeMode() -> bool;

    /**
     * Loading a const field
     */
    template <typename Field_ta>
    auto load(Field_ta&       field /**< the const field to be loaded            */,
              Neon::Compute   computeE = Neon::Compute::MAP /**< computation patter applied to the field */,
              StencilSemantic stencilSemantic = StencilSemantic::standard)
        -> std::enable_if_t<std::is_const_v<Field_ta>, const typename Field_ta::Partition&>;
};

}  // namespace Neon::set

#include "Neon/set/container/Loader_imp.h"