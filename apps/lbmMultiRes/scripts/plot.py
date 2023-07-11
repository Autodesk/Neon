import os
import numpy as np
import matplotlib.pyplot as plt


def plotme(dat_filename_X="", dat_filename_Y="", label="", png_filename="", scale=1):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)

    if (dat_filename_Y!=""):
        y_neon, u_neon = np.loadtxt(dat_filename_Y, unpack=True, usecols=(0, 1))
        axes.plot(y_neon, scale*u_neon, 'y.-', label=label)

        y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 1))
        axes.plot(y_ref, u_ref, 'bs', markerfacecolor='none', label='Ghia et al. 1982')

    if (dat_filename_X!=""):        
        x_neon, v_neon = np.loadtxt(dat_filename_X, unpack=True, usecols=(0, 1))
        axes.plot(x_neon, scale*v_neon, 'g.-', label=label)    

        x_ref, v_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 7))
        axes.plot(x_ref, v_ref, 'ks', markerfacecolor='none',
                  label='Ghia et al. 1982')

    axes.legend()
    axes.set_xlabel(r'Y')
    axes.set_ylabel(r'Velocity')
    plt.tight_layout()
    plt.savefig(png_filename)


plotme(dat_filename_X='NeonMultiResLBM_5000_X.dat',
       dat_filename_Y='NeonMultiResLBM_5000_Y.dat',
       label='Neon MultiRes LBM',
       png_filename='MultiResNeon_vs_ghia1982.png',
       scale=1)

plotme(dat_filename_Y='NeonUniformLBM_20000_Y.dat',
       label='Neon Uniform LBM',
       png_filename='UniformNeon_vs_ghia1982.png',
       scale=25)
