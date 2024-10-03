#pragma once
#include "parameters.h"

template <typename Parameters>
struct Fields
{
    using Field = typename Parameters::Field;
    using Grid = typename Parameters::Grid;
    using Type = typename Field::Type;

    static constexpr int spaceDim = Parameters::spaceDim;
    static constexpr int fieldCard = Parameters::fieldCard;

    Field fields[2];

    Fields(){

    }
    Fields(Grid & grid)
    {
        fields[0] = grid.template newField<Type, fieldCard>("u0", fieldCard, Type(0));
        fields[1] = grid.template newField<Type, fieldCard>("u1", fieldCard, Type(0));
    }

    auto get(int id) -> Field&
    {
        if (id == 0 || id == 1) {
            return fields[id];
        }
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
};