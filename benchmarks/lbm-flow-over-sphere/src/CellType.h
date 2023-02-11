#pragma once


struct CellType
{
    enum Classification : uint16_t
    {
        bounceBack = 1,
        movingWall = 2,
        bulk = 3,
        undefined = 4,
        pressure = 5,
        velocity = 6
    };

    /**
     * used to store 5 ids to identify knowns or middle sections
     */
    struct LatticeSectionUnk
    {
        unsigned int mA : 5;
        unsigned int mB : 5;
        unsigned int mC : 5;
        unsigned int mD : 5;
        unsigned int mE : 5;
    };

    struct LatticeSectionMiddle
    {
        unsigned int mA : 5;
        unsigned int mB : 5;
        unsigned int mC : 5;
        unsigned int mD : 5;
    };

    NEON_CUDA_HOST_DEVICE CellType(int dummy = 0)
    {
        (void)dummy;
        classification = bulk;
        wallNghBitflag = 0;
    }

    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c,
                                            uint32_t       n)
    {
        classification = c;
        wallNghBitflag = n;
    }

    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c)
    {
        classification = c;
        wallNghBitflag = 0;
    }


    uint32_t             wallNghBitflag;
    Classification       classification;
    LatticeSectionUnk    unknowns;
    LatticeSectionMiddle middle;
    float                rho;
    Neon::float_3d       u;
};

std::ostream& operator<<(std::ostream& os, const CellType& dt)
{
    os << static_cast<double>(dt.classification);
    return os;
}