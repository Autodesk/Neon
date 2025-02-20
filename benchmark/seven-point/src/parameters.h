#pragma once

template <int spaceDim_, int fieldCard_, typename Grid_, typename Type_>
struct Parameters
{
    using Type = Type_;
    using Field = typename Grid_::template Field<Type, fieldCard_>;
    using Grid = Grid_;
    static constexpr int spaceDim = spaceDim_;
    static constexpr int fieldCard = fieldCard_;
};

template <typename Parameters>
struct SevenPoint
{
    using Type = typename Parameters::Type;
    using Field = typename Parameters::Field;
    using Grid = typename Parameters::Grid;
    static constexpr int spaceDim = Parameters::spaceDim;
    static constexpr int fieldCard = Parameters::fieldCard;

    static constexpr int numNgh = Parameters::spaceDim;

    template <int direction, int components>
    static constexpr auto getOffset() -> int
    {
        if constexpr (spaceDim == 3) {
            if constexpr (direction == 0) {
                // only componend x (i.e. 0) does not have an offset of zero
                if constexpr (components == 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 1) {
                if constexpr (components == 0) {
                    return -1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 2) {
                if constexpr (components == 2) {
                    return 1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 3) {
                if constexpr (components == 2) {
                    return -1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 4) {
                if constexpr (components == 1) {
                    return 1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 5) {
                if constexpr (components == 1) {
                    return -1;
                } else {
                    return 0;
                }
            }
        }
        if constexpr (spaceDim == 2) {
            if constexpr (direction == 0) {
                // only componend x (i.e. 0) does not have an offset of zero
                if constexpr (components == 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 1) {
                // only componend x (i.e. 0) does not have an offset of zero
                if constexpr (components == 0) {
                    return -1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 2) {
                // only componend x (i.e. 0) does not have an offset of zero
                if constexpr (components == 2) {
                    return 1;
                } else {
                    return 0;
                }
            }
            if constexpr (direction == 3) {
                // only componend x (i.e. 0) does not have an offset of zero
                if constexpr (components == 2) {
                    return -1;
                } else {
                    return 0;
                }
            }
        }
    }
};
