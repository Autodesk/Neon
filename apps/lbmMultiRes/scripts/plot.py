import os
import numpy as np
import matplotlib.pyplot as plt

def plotme(dat_filename, label, png_filename, scale):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)

    y_neon, u_neon = np.loadtxt(dat_filename, unpack=True, usecols=(0, 1))
    p = y_neon.argsort()
    u_neon = u_neon[p]
    y_neon = y_neon[p]
    axes.plot(y_neon, scale*u_neon, 'b.', label=label)

    y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 1))
    axes.plot(y_ref, u_ref, 'ms', label='Ghia et al. 1982')

    axes.legend()
    axes.set_xlabel(r'Y')
    axes.set_ylabel(r'U')
    plt.tight_layout()
    plt.savefig(png_filename)

plotme('NeonMultiResLBM_5000.dat', 'Neon MultiRes LBM', 'MultiResNeon_vs_ghia1982', 1)
plotme('NeonUniformLBM_20000.dat', 'Neon Uniform LBM',  'UniformNeon_vs_ghia1982.png', 25)