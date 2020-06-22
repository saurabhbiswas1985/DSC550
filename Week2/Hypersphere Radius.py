# Assignment 2.1
# Hypersphere Radius
# Author: Saurabh Biswas
# DSC550 T302

# Import required libraries
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi


def hs_plot(d):
    '''This function accepts a list with dimensions and plot it against radius for volume 1'''

    hyper_rad = []  # empty list to hold hypersphere radius

    for n in d:     # run a loop for d dimension to get a list of radius for unit volume
        hyper_rad.append((((gamma(n/2+1))**(1/n))/pi**(1/2)*1**(1/2)))  # hypersphere volume is 1
    plt.plot(d, hyper_rad)  # plot a graph between dimension and volume
    plt.xlabel('Dimension')     # x-axis label
    plt.ylabel('Hypersphere Radius')    # y-axis label
    plt.title('Hypersphere Radius Plot')    # Title of the graph
    plt.show()  # show the plot


# invoke main program
if __name__ == '__main__':
    rad = []  # create an empty list of dimensions
    for x in range(1, 101):  # create a list of dimension from 1 to 100
        rad.append(x)

    hs_plot(rad)    # invoke the function to plot dimension v/s radius graph
