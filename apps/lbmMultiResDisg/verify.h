#pragma once
#include <algorithm>
#include <array>
#include <map>
#include <vector>

namespace detail {
//Table 1 and Table 2 from
// U Ghia, K.N Ghia, C.T Shin,
// High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method,
// Journal of Computational Physics,
// Volume 48, Issue 3,
// 1982,
// Pages 387-411,
// ISSN 0021-9991,
// https://doi.org/10.1016/0021-9991(82)90058-4

constexpr int ghiaNumPoints = 17;

std::array<float, ghiaNumPoints> ghiaYPos = {0.000000, 0.054700, 0.062500, 0.070300, 0.101600, 0.171900, 0.281300, 0.453100, 0.500000, 0.617200, 0.734400, 0.851600, 0.953100, 0.960900, 0.968800, 0.976600, 1.000000};

std::array<float, ghiaNumPoints> ghiaXPos = {0.000000, 0.062500, 0.070300, 0.078100, 0.093800, 0.156300, 0.226600, 0.234400, 0.500000, 0.804700, 0.859400, 0.906300, 0.945300, 0.953100, 0.960900, 0.968800, 1.000000};


std::map<int, std::array<float, ghiaNumPoints>> ghiaYVals{
    {100, {0.000000, -0.037170, -0.041920, -0.047750, -0.064340, -0.101500, -0.156620, -0.210900, -0.205810, -0.136410, 0.003320, 0.231510, 0.687170, 0.737220, 0.788710, 0.841230, 1.000000}},
    {1000, {0.000000, -0.181090, -0.201960, -0.222200, -0.297300, -0.382890, -0.278050, -0.106480, -0.060800, 0.057020, 0.187190, 0.333040, 0.466040, 0.511170, 0.574920, 0.659280, 1.000000}},
    {3200, {0.000000, -0.32407, -0.35344, -0.37827, -0.41933, -0.34323, -0.24427, -0.86636, -0.04272, 0.071560, 0.197910, 0.346820, 0.461010, 0.465470, 0.482960, 0.532360, 1.000000}},
    {5000, {0.000000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855, -0.07404, -0.03039, 0.081830, 0.200870, 0.335560, 0.460360, 0.459920, 0.461200, 0.482230, 1.000000}},
    {10000, {0.000000, -0.42735, -0.42537, -0.41657, -0.38000, -0.32709, -0.23186, -0.07540, 0.031110, 0.083440, 0.206730, 0.346350, 0.478040, 0.480700, 0.477830, 0.472210, 1.000000}}};


std::map<int, std::array<float, ghiaNumPoints>> ghiaXVals{
    {100, {0.000000, 0.092330, 0.100910, 0.108900, 0.123170, 0.160770, 0.175070, 0.175270, 0.054540, -0.245330, -0.224450, -0.169140, -0.103130, -0.088640, -0.073910, -0.059060, 0.000000}},
    {1000, {0.000000, 0.274850, 0.290120, 0.303530, 0.326270, 0.370950, 0.330750, 0.322350, 0.024260, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.000000}},
    {3200, {0.000000, 0.395600, 0.409170, 0.419060, 0.427680, 0.371190, 0.290300, 0.281880, 0.009990, -0.31184, -0.37401, -0.44307, -0.54053, -0.52357, -0.47425, -0.39017, 0.000000}},
    {5000, {0.000000, 0.424470, 0.433290, 0.436480, 0.429510, 0.353680, 0.280660, 0.272800, 0.009450, -0.30018, -0.36214, -0.41442, -0.52876, -0.55408, -0.55069, -0.49774, 0.000000}},
    {10000, {0.00000, 0.43983, 0.43733, 0.43124, 0.41487, 0.35070, 0.28003, 0.27224, 0.00831, -0.30719, -0.36737, -0.41496, -0.45863, -0.49099, -0.52987, -0.54302, 0.000000}}};


}  // namespace detail

template <typename T>
inline T verifyGhia1982(const int                           Re,
                        const std::vector<std::pair<T, T>>& xPosVal,
                        const std::vector<std::pair<T, T>>& yPosVal)
{
    //we assume the ghia points are far less than the input points. so, for every ghia point, we try to find the interval in which it lies in the input points
    //then linearly interpolate the values between the ends of this interval, compute the different between the interpolated value and ghia value. Finally, we
    //report the max different. Note that report the l2 norm of the difference could be also used for grid refinement analysis

    auto calcDiff = [&](const std::vector<std::pair<T, T>>&             posVal,
                        const std::array<float, detail::ghiaNumPoints>& ghiaPos,
                        const std::array<float, detail::ghiaNumPoints>& ghiaVal) {
        std::array<T, detail::ghiaNumPoints> diff;

        for (size_t i = 0; i < ghiaPos.size(); ++i) {
            const float pos = static_cast<T>(ghiaPos[i]);
            const float val = static_cast<T>(ghiaVal[i]);

            const auto itr = std::lower_bound(posVal.begin(), posVal.end(), pos, [=](const std::pair<T, T>& a, const T& b) { return a.first < b; });

            const size_t low = (itr == posVal.end()) ? posVal.size() - 1 : itr - posVal.begin();
            const size_t high = (low >= posVal.size() - 1 || itr == posVal.end()) ? low : low + 1;

            const T lowPos = posVal[low].first;
            const T highPos = posVal[high].first;

            const T lowVal = posVal[low].second;
            const T highVal = posVal[high].second;

            T interp;
            if (low == high) {
                interp = lowVal;
            } else {
                interp = lowVal + ((pos - lowPos) / (highPos - lowPos)) * (highVal - lowVal);
            }

            diff[i] = std::abs(interp - val) / (val == 0.0 ? 1 : val);
        }
        return diff;
    };


    auto difX = calcDiff(xPosVal, detail::ghiaXPos, detail::ghiaXVals[Re]);
    auto difY = calcDiff(yPosVal, detail::ghiaYPos, detail::ghiaYVals[Re]);

    T maxDiffX = *std::max_element(difX.begin(), difX.end());
    T maxDiffY = *std::max_element(difY.begin(), difY.end());

    return std::max(maxDiffX, maxDiffY);
}