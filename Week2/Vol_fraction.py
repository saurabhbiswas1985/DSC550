# Assignment 2.1
# Fraction of Volume
# Author: Saurabh Biswas
# DSC550 T302

# Import required libraries
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi


def fraction_volume(dim, len, rad):
    ''' This function calculate fraction volume'''
    frac_vol_list = []         # List of Fraction of volume for each d

    for n in dim:
        hypers_vol = (((pi ** (n / 2)) / gamma(n / 2 + 1)) * rad ** n)  # vol of hypersphere
        hyperc_vol = len ** n    # vol of hypercube
        frac_vol = hypers_vol/hyperc_vol   # Calculate fraction of volume
        frac_vol_list.append(frac_vol)

    return frac_vol_list


def plot_function(dim, frac_vol, xlbl, ylbl, plt_title):
    ''' This function plots dimension v/s fraction volume'''

    plt.plot(dim, frac_vol)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(plt_title)
    plt.show()


def fraction_shell(dim, len, e):
    '''This function calculates the volume of thin shell'''

    frac_shell_vol_list = []

    for n in dim:
        hyperc_out_vol = len ** n   # outer cube volume
        hyperc_in_vol = (len - e) ** n  # inner cube volume
        frac_shell_vol_list.append((hyperc_out_vol-hyperc_in_vol)/hyperc_out_vol)  # Frac vol of thin shell

    return frac_shell_vol_list


if __name__ == '__main__':

    len = 2     # length of the hypercube
    rad = 2/2   # radius of the hypersphere
    dim = [i for i in range(1, 101)]  # Create list of dimension values 1-100

    frac_vol = fraction_volume(dim, len, rad)
    xlbl = 'Dimension'
    ylbl = 'Fraction of Volume'
    plt_title = 'Fraction of Volume V/S Dimension'
    plot_function(dim, frac_vol, xlbl, ylbl, plt_title)

    e = 0.01
    dim = [n for n in range(1, 1000, 1)]     # dimension
    frac_shell_vol_list = fraction_shell(dim, len, e)  # Get the fraction volume of thin shell
    xlbl = 'Dimension'
    ylbl = 'Fraction of Volume of Shell'
    plt_title = 'Fraction of Volume of Shell V/S Dimension'
    plot_function(dim, frac_shell_vol_list, xlbl, ylbl, plt_title)

