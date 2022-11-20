struct CellType
{
    enum Classification : int
    {
        bounceBack,
        bulk
    };

    NEON_CUDA_HOST_DEVICE CellType(int dummy = 0)
    {
        (void)dummy;
        classification = bulk;
        bounceBackNghBitflag = 0;
    }

    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c, uint32_t n)
    {
        classification = c;
        bounceBackNghBitflag = n;
    }
    NEON_CUDA_HOST_DEVICE explicit CellType(Classification c)
    {
        classification = c;
        bounceBackNghBitflag = 0;
    }

    uint32_t       bounceBackNghBitflag;
    Classification classification;
};