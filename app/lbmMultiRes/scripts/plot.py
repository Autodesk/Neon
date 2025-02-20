#https://help.altair.com/hwcfdsolvers/nfx/topics/nanofluidx/lid_driven_cavity_2d_r.htm
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = ['Palatino Linotype', 'serif']
rcParams['font.size'] = 10
rcParams["mathtext.default"] = 'regular'

def plotme(dat_filename_X="", dat_filename_Y="", label="", png_filename="", scale=1):

    fig, ax1  = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=1000)

    if (dat_filename_X!=""):        
        x_neon, v_neon = np.loadtxt(dat_filename_X, unpack=True, usecols=(0, 1))
        x_neon -= 0.5
        x_neon *= 2.0        
        ax1.plot(x_neon, scale*v_neon, 'k-', label=label)    

        x_ref, v_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(6, 7))
        x_ref -= 0.5
        x_ref *= 2.0
        ax1.plot(x_ref, v_ref, 'o', markerfacecolor='none',
                  label='Ghia 1982')
        
        ax1.set_xlabel(r'x')
        ax1.set_ylabel('v/u$_{lid}$')
        ax1.legend(loc = "upper left")
        
    if (dat_filename_Y!=""):    
        ax2 = ax1 .twinx()
        ax3 = ax1 .twiny()

        y_neon, u_neon = np.loadtxt(dat_filename_Y, unpack=True, usecols=(0, 1))
        y_neon -=0.5
        y_neon *= 2.0
        ax2.plot(scale*u_neon, y_neon, 'k-', label=label)
        
        ax3.set_xlim(ax2.get_ylim())  

        y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 1))
        y_ref -=0.5
        y_ref *= 2.0
        ax3.plot(u_ref, y_ref, 'o', markerfacecolor='none', label='Ghia 1982')
        
        ax2.set_ylabel('u/u$_{lid}$')
        ax3.set_xlabel(r'y')

        #h1, l1 = ax1.get_legend_handles_labels()
        #h2, l2 = ax2.get_legend_handles_labels()
        #h3, l3 = ax3.get_legend_handles_labels()
        #ax1.legend(h1+h2+h3, l1+l2+l3, loc=2)

    
    plt.tight_layout()
    plt.savefig(png_filename)


plotme(dat_filename_X='NeonMultiResLBM_5000_X.dat',
       dat_filename_Y='NeonMultiResLBM_5000_Y.dat',
       label='Ours',
       png_filename='MultiResNeon_vs_ghia1982.png',
       scale=1)

plotme(dat_filename_Y='NeonUniformLBM_20000_Y.dat',
       label='Uniform LBM',
       png_filename='UniformNeon_vs_ghia1982.png',
       scale=25)
