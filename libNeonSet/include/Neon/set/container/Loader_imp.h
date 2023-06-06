#pragma once

#include "type_traits"

#include "Neon/set/Containter.h"

namespace Neon::set {
namespace internal {

namespace tmp {
// From:
// https://en.cppreference.com/w/cpp/experimental/is_detected
namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type = Op<Args...>;
};

}  // namespace detail

struct nonesuch
{
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

template <template <class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

}  // namespace tmp

template <typename Field_ta>
struct DataTransferExtractor
{
    // field.haloUpdate(bk, opt);
    // const Neon::set::Backend& /*bk*/,
    // Neon::set::HuOptions_t& /*opt*/
   private:
    template <typename T>
    using HaloUpdate = decltype(std::declval<T>().newHaloUpdate(std::declval<Neon::set::StencilSemantic>(),
                                                                std::declval<Neon::set::TransferMode>(),
                                                                std::declval<Neon::Execution>()));

    template <typename T>
    static constexpr bool HasHaloUpdateMethod = tmp::is_detected_v<HaloUpdate, T>;

   public:
    /**
     * Function that extract that wraps the haloUpdateContainer method of a field for future use.
     * If the function is not detected the input parameter status flag is set to false.
     */
    static auto get([[maybe_unused]] const Field_ta& field /**< target field */,
                    bool&                            status /**< status flag. True means extraction was successful */) -> auto
    {
        if constexpr (HasHaloUpdateMethod<Field_ta>) {
            auto huFun = [field, &status](Neon::set::TransferMode    transferMode,
                                          Neon::set::StencilSemantic stencilSemantic)
                -> Neon::set::Container {
                Neon::set::Container container = field.newHaloUpdate(stencilSemantic, transferMode, Neon::Execution::device);
                return container;
            };
            status = true;
            return huFun;
        } else {
            auto huFun = [field, &status](Neon::set::TransferMode    transferMode,
                                          Neon::set::StencilSemantic stencilSemantic)
                -> Neon::set::Container {
                (void)transferMode;
                (void)stencilSemantic;
                return {};
            };
            status = false;
            return huFun;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
};


}  // namespace internal

template <typename Field_ta>
auto Loader::
    load(Field_ta&       field,
         Neon::Pattern   computeE,
         StencilSemantic stencilSemantic)
        -> std::enable_if_t<!std::is_const_v<Field_ta>, typename Field_ta::Partition&>
{
    switch (m_loadingMode) {
        case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
            using namespace Neon::set::dataDependency;
            Neon::set::dataDependency::MultiXpuDataUid uid = field.getUid();
            constexpr auto                             access = Neon::set::dataDependency::AccessType::WRITE;
            Pattern                                    compute = computeE;
            Token                                      token(uid, access, compute);

            if (compute == Neon::Pattern::STENCIL &&
                (stencilSemantic == StencilSemantic::standard ||
                 stencilSemantic == StencilSemantic::streaming)) {
                Neon::NeonException exp("Loader");
                exp << "Loading a non const field for a stencil operation is not supported in Neon";
                NEON_THROW(exp);
            }

            m_container.addToken(token);

            return field.getPartition(mExecution, m_setIdx, m_dataView);
        }
        case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
            return field.getPartition(mExecution, m_setIdx, m_dataView);
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

/**
 * Loading a const field
 */
template <typename Field_ta>
auto Loader::
    load(Field_ta&       field,
         Neon::Pattern   computeE,
         StencilSemantic stencilSemantic)
        -> std::enable_if_t<std::is_const_v<Field_ta>, const typename Field_ta::Partition&>
{
    switch (m_loadingMode) {
        case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
            using namespace Neon::set::dataDependency;
            Neon::set::dataDependency::MultiXpuDataUid uid = field.getUid();
            constexpr auto                             access = Neon::set::dataDependency::AccessType::READ;
            Neon::Pattern                              compute = computeE;
            Token                                      token(uid, access, compute);

            if (compute == Neon::Pattern::STENCIL) {
                token.setDataTransferContainer(
                    [&, stencilSemantic](Neon::set::TransferMode transferMode)
                        -> Neon::set::Container {
                        // TODO: add back following line with template metaprogramming
                        // field.haloUpdate(bk, opt);
                        // https://gist.github.com/fenbf/d2cd670704b82e2ce7fd
                        bool status;
                        auto huFun = internal::DataTransferExtractor<Field_ta>::get(field, status);

                        if (!status) {
                            Neon::NeonException e("Neon::Loader");
                            e << std::string("Unable to extract haloUpdateContainer from the target field: ") + field.getName() + " UID " + std::to_string(field.getUid());
                            NEON_THROW(e);
                        }

                        Neon::set::Container container = huFun(transferMode, stencilSemantic);

                        if (container.getContainerInterface().getContainerExecutionType() != ContainerExecutionType::graph) {
                            NEON_THROW_UNSUPPORTED_OPERATION("Halo update Containers type should be Graph");
                        }

                        return container;
                    });
            }
            m_container.addToken(token);

            return field.getPartition(mExecution, m_setIdx, m_dataView);
        }
        case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
            return field.getPartition(mExecution, m_setIdx, m_dataView);
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::set