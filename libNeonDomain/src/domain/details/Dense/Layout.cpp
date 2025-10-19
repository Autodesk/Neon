/**
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 *                           DENSE GRID LAYOUT IMPLEMENTATION
 *                      LayoutUtils and CLI Functionality for Dense Grids
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 *
 * This file implements the complete Layout utility suite for Dense Grid memory layouts.
 * It provides:
 *
 * • String conversion utilities (toString, fromString)
 * • Integer conversion utilities (toInt, fromInt) 
 * • Layout introspection (isDisgSoA, isSoA, isAoS)
 * • Complete CLI integration with validation and error reporting
 * • Integration with Neon's reporting system
 *
 * The implementation follows the same patterns as other Neon utilities like
 * StencilSemanticUtils::Cli, ensuring consistency across the framework.
 *
 * ═══════════════════════════════════════════════════════════════════════════════════════
 */

#include "Neon/domain/details/Dense/Layout.h"

namespace Neon::domain::details::Dense {

// ═══════════════════════════════════════════════════════════════════════════════════════
// STRING CONVERSION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════════════

auto LayoutUtils::toString(const Layout& layout) -> std::string
{
    switch (layout) {
        case Layout::StructOfArrays: {
            return "StructOfArrays";
        }
        case Layout::ArrayOfStructs: {
            return "ArrayOfStructs";
        }
        case Layout::DisgSoA: {
            return "DisgSoA";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto LayoutUtils::fromString(const std::string& layout) -> Layout
{
    std::array<Layout, 3> opts{Layout::StructOfArrays, Layout::ArrayOfStructs, Layout::DisgSoA};
    for (auto a : opts) {
        if (toString(a) == layout) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// INTEGER CONVERSION IMPLEMENTATION  
// ═══════════════════════════════════════════════════════════════════════════════════════

auto LayoutUtils::toInt(const Layout& layout) -> int
{
    return static_cast<int>(layout);
}

auto LayoutUtils::fromInt(const int& layout) -> Layout
{
    switch (layout) {
        case 0:
            return Layout::StructOfArrays;
        case 1:
            return Layout::ArrayOfStructs;
        case 2:
            return Layout::DisgSoA;
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// INTROSPECTION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════════════

auto LayoutUtils::getOptions() -> std::array<Layout, nOptions>
{
    std::array<Layout, nOptions> opts = {Layout::StructOfArrays, Layout::ArrayOfStructs, Layout::DisgSoA};
    return opts;
}

auto LayoutUtils::isDisgSoA(const Layout& layout) -> bool
{
    return layout == Layout::DisgSoA;
}

auto LayoutUtils::isSoA(const Layout& layout) -> bool
{
    return layout == Layout::StructOfArrays;
}

auto LayoutUtils::isAoS(const Layout& layout) -> bool
{
    return layout == Layout::ArrayOfStructs;
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// COMMAND-LINE INTERFACE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════════════
LayoutUtils::Cli::Cli()
{
    mSet = false;
}

LayoutUtils::Cli::Cli(std::string s)
{
    set(s);
}

LayoutUtils::Cli::Cli(Layout layout)
{
    mOption = layout;
    mSet = true;
}

auto LayoutUtils::Cli::getOption() const -> Layout
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Layout was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return mOption;
}

auto LayoutUtils::Cli::set(const std::string& opt) -> void
{
    try {
        mOption = LayoutUtils::fromString(opt);
    } catch (...) {
        std::stringstream errorMsg;
        errorMsg << "Layout: " << opt << " is not a valid option (valid options are {";
        auto options = LayoutUtils::getOptions();
        int  i = 0;
        for (auto o : options) {
            if (i != 0) {
                errorMsg << ", ";
            }
            errorMsg << LayoutUtils::toString(o);
            i = 1;
        }
        errorMsg << "})";
        NEON_ERROR(errorMsg.str());
    }
    mSet = true;
}

auto LayoutUtils::Cli::getStringOptions() const -> std::string
{
    std::stringstream s;
    auto              options = LayoutUtils::getOptions();
    int               i = 0;
    for (auto o : options) {
        if (i != 0) {
            s << ", ";
        }
        s << LayoutUtils::toString(o);
        i = 1;
    }
    std::string msg = s.str();
    return msg;
}

auto LayoutUtils::Cli::getStringOption() const -> std::string
{
    if (!mSet) {
        std::stringstream errorMsg;
        errorMsg << "Layout was not set.";
        NEON_ERROR(errorMsg.str());
    }
    return LayoutUtils::toString(mOption);
}

auto LayoutUtils::Cli::getDoc() const -> std::string
{
    std::stringstream s;
    s << getStringOptions();
    s << " default: " << LayoutUtils::toString(Layout::StructOfArrays);
    return s.str();
}

// ═══════════════════════════════════════════════════════════════════════════════════════
// NEON REPORTING SYSTEM INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════════════

#if !defined(NEON_WARP_COMPILATION)
auto LayoutUtils::Cli::addToReport(Neon::Report& report) const -> void
{
    report.addMember("Layout", LayoutUtils::toString(this->getOption()));
}

auto LayoutUtils::Cli::addToReport(Neon::Report& report, Neon::Report::SubBlock& subBlock) const -> void
{
    report.addMember("Layout", LayoutUtils::toString(this->getOption()), &subBlock);
}
#endif

}  // namespace Neon::domain::details::Dense
