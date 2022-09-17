#pragma once
// #include <experimental/type_traits>
#include "Neon/set/DevSet.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/dependency/Token.h"

#include "type_traits"
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
struct HaloUpdateExtractor_t
{
    // field.haloUpdate(bk, opt);
    // const Neon::set::Backend_t& /*bk*/,
    // Neon::set::HuOptions_t& /*opt*/
   private:
    template <typename T>
    using HaloUpdate = decltype(std::declval<T>().haloUpdate(std::declval<Neon::set::HuOptions&>()));

    template <typename T>
    static constexpr bool HasHaloUpdate = tmp::is_detected_v<HaloUpdate, T>;

   public:
    static auto get([[maybe_unused]] const Field_ta& field) -> auto
    {
        if constexpr (HasHaloUpdate<Field_ta>) {
            auto huFun = [field](Neon::set::HuOptions& opt) {
                field.haloUpdate(opt);
            };
            return huFun;
        } else {
            auto huFun = [field](Neon::set::HuOptions& opt) {
                (void)opt;
            };

            return huFun;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
};

template <typename Field_ta>
struct HaloUpdatePerDeviceExtractor_t
{
    // field.haloUpdate(bk, opt);
    // const Neon::set::Backend_t& /*bk*/,
    // Neon::set::HuOptions_t& /*opt*/
   private:
    template <typename T>
    using HaloUpdatePerDevice = decltype(std::declval<T>().haloUpdate(std::declval<Neon::SetIdx&>(),
                                                                      std::declval<Neon::set::HuOptions&>()));

    template <typename T>
    static constexpr bool HasHaloUpdatePerDevice = tmp::is_detected_v<HaloUpdatePerDevice, T>;

   public:
    static auto get([[maybe_unused]] const Field_ta& field) -> auto
    {
        if constexpr (HasHaloUpdatePerDevice<Field_ta>) {
            auto huFun = [field](Neon::SetIdx          setIdx,
                                 Neon::set::HuOptions& opt) {
                field.haloUpdate(setIdx, opt);
            };
            return huFun;
        } else {
            auto huFun = [field](Neon::SetIdx          setIdx,
                                 Neon::set::HuOptions& opt) {
                (void)opt;
                (void)setIdx;
            };

            return huFun;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
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

    Neon::DeviceType                      m_devE;
    Neon::SetIdx                          m_setIdx;
    Neon::DataView                        m_dataView;
    Neon::set::internal::LoadingMode_e::e m_loadingMode;

   public:
    Loader(Neon::set::internal::ContainerAPI&    container,
           Neon::DeviceType                      devE,
           Neon::SetIdx                          setIdx,
           Neon::DataView                        dataView,
           Neon::set::internal::LoadingMode_e::e loadingMode)
        : m_container(container),
          m_devE(devE),
          m_setIdx(setIdx),
          m_dataView(dataView),
          m_loadingMode(loadingMode)
    {
    }

   private:
   public:
    enum struct StencilOptions
    {
        LATTICE,
        LATTICE_REVERSED,
        DEFAULT,
        DEFAULT_REVERSE,
    };


    template <typename Field_ta>
    auto load(Field_ta&      field,
              Neon::Compute  computeE = Neon::Compute::MAP,
              StencilOptions stencilOptions = StencilOptions::DEFAULT)
        -> std::enable_if_t<!std::is_const_v<Field_ta>, typename Field_ta::Partition&>
    {

        switch (m_loadingMode) {
            case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
                Neon::internal::dataDependency::DataUId              uid = field.getUid();
                constexpr Neon::internal::dataDependency::AccessType access = Neon::internal::dataDependency::AccessType::WRITE;
                Compute                                              compute = computeE;
                Neon::internal::dataDependency::Token                dataToken(uid, access, compute);

                if (compute == Neon::Compute::STENCIL &&
                    (stencilOptions == StencilOptions::DEFAULT || stencilOptions == StencilOptions::LATTICE)) {
                    Neon::NeonException exp("Loader");
                    exp << "Loading a non const field for a stencil operation is not supported in Neon";
                    NEON_THROW(exp);
                }

                m_container.addToken(dataToken);

                return field.getPartition(m_devE, m_setIdx, m_dataView);
            }
            case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
                return field.getPartition(m_devE, m_setIdx, m_dataView);
            }
        }
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    auto computeMode() -> bool
    {
        return m_loadingMode == Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA;
    }

    /**
     * Loading a const field
     */
    template <typename Field_ta>
    auto load(Field_ta&     field /**< the const field to be loaded            */,
              Neon::Compute computeE = Neon::Compute::MAP /**< computation patter applied to the field */)
        -> std::enable_if_t<std::is_const_v<Field_ta>, const typename Field_ta::Partition&>
    {
        switch (m_loadingMode) {
            case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
                Neon::internal::dataDependency::DataUId              uid = field.getUid();
                constexpr Neon::internal::dataDependency::AccessType access = Neon::internal::dataDependency::AccessType::READ;
                Neon::Compute                                        compute = computeE;
                Neon::internal::dataDependency::Token                dataToken(uid, access, compute);

                if (compute == Neon::Compute::STENCIL) {
                    dataToken.setHaloUpdate(
                        [&](Neon::set::HuOptions& opt) -> void {
                            // TODO: add back following line with template metaprogramming
                            // field.haloUpdate(bk, opt);
                            // https://gist.github.com/fenbf/d2cd670704b82e2ce7fd
                            auto huFun = internal::HaloUpdateExtractor_t<Field_ta>::get(field);
                            huFun( opt);

                        return; },
                        [&](Neon::SetIdx setIdx, Neon::set::HuOptions& opt) -> void {
                            auto huFun = internal::HaloUpdatePerDeviceExtractor_t<Field_ta>::get(field);
                            huFun(setIdx, opt);

                            return;
                        });
                }
                m_container.addToken(dataToken);

                return field.getPartition(m_devE, m_setIdx, m_dataView);
            }
            case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
                return field.getPartition(m_devE, m_setIdx, m_dataView);
            }
        }
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
};

}  // namespace Neon::set