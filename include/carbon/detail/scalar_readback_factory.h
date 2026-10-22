#pragma once

#include <carbon/reduce.h>

#include <memory>

namespace carbon {

class Kernel;

namespace detail {

class ScalarReadbackFactoryBase {
public:
  ScalarReadbackFactoryBase() = default;
  ScalarReadbackFactoryBase(const ScalarReadbackFactoryBase&) = delete;
  ScalarReadbackFactoryBase(ScalarReadbackFactoryBase&&) = delete;
  auto
  operator=(const ScalarReadbackFactoryBase&) -> ScalarReadbackFactoryBase& = delete;
  auto
  operator=(ScalarReadbackFactoryBase&&) -> ScalarReadbackFactoryBase& = delete;
  virtual ~ScalarReadbackFactoryBase() = default;

  [[nodiscard]] virtual auto
  make() const -> Kernel = 0;
};

template <typename T>
class TypedScalarReadbackFactory final : public ScalarReadbackFactoryBase {
public:
  explicit TypedScalarReadbackFactory(ReductionResult<T>& dst) noexcept
      : dst_(&dst) {}

  [[nodiscard]] auto
  make() const -> Kernel override;

private:
  ReductionResult<T>* dst_{nullptr};
};

}  // namespace detail
}  // namespace carbon
