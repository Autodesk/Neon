#include "Neon/domain/tools/TestData.h"

namespace Neon::domain::tool::testing {

auto FieldNamesUtils::toString(FieldNames name) -> std::string
{
    switch (name) {
        case FieldNames::X:
            return "X";
        case FieldNames::Y:
            return "Y";
        case FieldNames::Z:
            return "Z";
        case FieldNames::W:
            return "W";
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto FieldNamesUtils::toInt(FieldNames name) -> int
{
    return static_cast<int>(name);
}

auto FieldNamesUtils::fromInt(int id) -> FieldNames
{
    if (toInt(FieldNames::X) == id)
        return FieldNames::X;
    if (toInt(FieldNames::Y) == id)
        return FieldNames::Y;
    if (toInt(FieldNames::Z) == id)
        return FieldNames::Z;
    if (toInt(FieldNames::W) == id)
        return FieldNames::W;

    NEON_THROW_UNSUPPORTED_OPTION("");
}

}  // namespace Neon::grid::tool::testing
