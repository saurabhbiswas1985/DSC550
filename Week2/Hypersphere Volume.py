# Assignment 2.1
# Hypersphere Volume
# Author: Saurabh Biswas
# DSC550 T302

# Import required libraries
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi


def hs_plot(d):
    '''This function accepts a list with dimensions and plot it against volume'''

    hyper_vol = []  # empty list to hold hypersphere volume

    for n in d:     # run a loop for d dimension to get a list of volumes
        hyper_vol.append(((pi**(n/2))/gamma(n/2+1))*1**n)   # for unit hypersphere radius is 1

    plt.plot(d, hyper_vol)  # plot a graph between dimension and volume
    plt.xlabel('Dimension')     # x-axis label
    plt.ylabel('Hypersphere Volume')    # y-axis label
    plt.title('Hypersphere Volume Plot')    # Title of the graph
    plt.show()  # show the plot


# invoke main program
if __name__ == '__main__':
    dim = []  # create an empty list of dimensions
    for x in range(1, 51):  # create a list of dimension from 1 to 50
        dim.append(x)

    hs_plot(dim)    # invoke the function to plot dimension v/s volume graph
