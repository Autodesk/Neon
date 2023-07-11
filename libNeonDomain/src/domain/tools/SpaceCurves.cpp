#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/core/types/Exceptions.h"

namespace Neon::domain::tool::spaceCurves {

auto EncoderTypeUtil::validOptions() -> std::array<EncoderType, EncoderTypeUtil::nConfig>
{
    std::array<EncoderType, EncoderTypeUtil::nConfig> options = {EncoderType::sweep,
                                                                       EncoderType::morton,
                                                                       EncoderType::hilbert};
    return options;
}

auto EncoderTypeUtil::toString(EncoderType e) -> std::string
{
    switch (e) {
        case EncoderType::sweep: {
            return "sweep";
        }
        case EncoderType::morton: {
            return "morton";
        }
        case EncoderType::hilbert: {
            return "hilbert";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("EncoderTypeUtil");
        }
    }
}

auto EncoderTypeUtil::fromInt(int val) -> EncoderType
{
    switch (val) {
        case static_cast<int>(EncoderType::sweep): {
            return EncoderType::sweep;
        }
        case static_cast<int>(EncoderType::morton): {
            return EncoderType::morton;
        }
        case static_cast<int>(EncoderType::hilbert): {
            return EncoderType::hilbert;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("EncoderTypeUtil");
        }
    }
}

auto EncoderTypeUtil::toInt(EncoderType dataView) -> int
{
    return static_cast<int>(dataView);
}

std::ostream& operator<<(std::ostream& os, EncoderType const& m)
{
    return os << std::string(EncoderTypeUtil::toString(m));
}

}  // namespace Neon::domain::tool::spaceCurves
