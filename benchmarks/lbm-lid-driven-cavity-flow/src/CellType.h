#pragma once

struct CellType
{
    enum Classification : int
    {
        bounceBack,
        movingWall,
        bulk,
        undefined
    };

    NEON_CUDA_HOST_DEVICE CellType(int dummy = 0)
    {
        (void)dummy;
        classification = bulk;
        wallNghBitflag = 0;
    }

    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c, uint32_t n)
    {
        classification = c;
        wallNghBitflag = n;
    }

    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c)
    {
        classification = c;
        wallNghBitflag = 0;
    }

    // Converting to int to exportVti
    operator int() const { return int(classification); }

    template <int fwdRegIdx>
    static auto isWall(const uint32_t& wallNghBitFlag)
        -> bool
    {
        return wallNghBitFlag & (uint32_t(1) << fwdRegIdx);
    }

    auto setWall(int fwdRegIdx)
        -> void
    {
        wallNghBitflag = wallNghBitflag | ((uint32_t(1) << fwdRegIdx));
    }

    uint32_t       wallNghBitflag;
    Classification classification;
};

std::ostream& operator<<(std::ostream& os, const CellType& dt)
{
    os << static_cast<double>(dt.classification);
    return os;
}