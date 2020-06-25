# Assignment 2.1
# Nearest Neighbour
# Author: Saurabh Biswas
# DSC550 T302

#   Import required libraries
import numpy as np
import random
import matplotlib.pyplot as plt


def random_number(d):
    '''accepts dimension as an argument and generates 10000 random number between 0 to 1
    for n dimension
    '''
    random_points = []  # create an empty list for random points
    for i in range(10000):
        random_points.append(np.random.uniform(low=0, high=1, size=d))  # generate uniform random number in d dimension

    return random_points


def calc_dist(d):
    '''This function accept dimension and calculate distance from center'''
    dist_list = []   # empty list
    random_points = random_number(d)
    for i in range(10000):
        x = 0

        for j in range(d):
            x += (0.5-random_points[i][j]) ** 2
        dist = x ** 0.5     # euclidean distance formula
        dist_list.append(dist)  # append into a list

    return max(dist_list), min(dist_list), min(dist_list)/max(dist_list)


def plot_graph(d,y, xlbl, ylbl, title):
    ''' This function plots dimension vs Y'''
    plt.plot(d, y)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    dist_max = []   # maximum distance list
    dist_min = []   # min distance list
    dist_ratio = []     # min/max ratio
    dim = []    # dimension list

    for d in range(1, 100):
        max_dist, min_dist, ratio_dist = calc_dist(d)
        dist_max.append(max_dist)
        dist_min.append(min_dist)
        dist_ratio.append(ratio_dist)
        dim.append(d)

    xlbl = 'Dimension'

    ylbl = 'Max Distance'
    title = 'Dimension v/s Max Distance'
    plot_graph(dim, dist_max, xlbl, ylbl, title)

    ylbl = 'Min Distance'
    title = 'Dimension v/s Min Distance'
    plot_graph(dim, dist_min, xlbl, ylbl, title)

    ylbl = 'Distance Ratio'
    title = 'Dimension v/s Distance Ratio (min/max)'
    plot_graph(dim, dist_ratio, xlbl, ylbl, title)
