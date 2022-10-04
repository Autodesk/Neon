#include "Neon/core/types/DataView.h"
#include "Neon/core/types/Exceptions.h"
namespace Neon {


auto DataViewUtil::validOptions() -> std::array<Neon::DataView, DataViewUtil::nConfig>
{
    std::array<Neon::DataView, DataViewUtil::nConfig> options = {DataView::STANDARD,
                                                                 DataView::INTERNAL,
                                                                 DataView::BOUNDARY};
    return options;
}

auto DataViewUtil::toString(DataView e) -> std::string
{
    switch (e) {
        case DataView::STANDARD: {
            return "STANDARD";
        }
        case DataView::INTERNAL: {
            return "INTERNAL";
        }
        case DataView::BOUNDARY: {
            return "BOUNDARY";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("DataViewUtil");
        }
    }
}

auto DataViewUtil::fromInt(int val) -> DataView
{
    switch (val) {
        case static_cast<int>(DataView::STANDARD): {
            return DataView::STANDARD;
        }
        case static_cast<int>(DataView::INTERNAL): {
            return DataView::INTERNAL;
        }
        case static_cast<int>(DataView::BOUNDARY): {
            return DataView::BOUNDARY;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("DataViewUtil");
        }
    }
}

auto DataViewUtil::toInt(DataView dataView) -> int
{
    return static_cast<int>(dataView);
}

std::ostream& operator<<(std::ostream& os, Neon::DataView const& m)
{
    return os << std::string(Neon::DataViewUtil::toString(m));
}

}  // namespace Neon
