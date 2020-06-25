# Assignment 2.1
# High Dimensional Data Analysis
# Author: Saurabh Biswas
# DSC550 T302

# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF


def cal_angle(d):
    '''This function returns a list of 100000 angles for a given dimension '''

    angle_list = []     # empty list

    for i in range(100000):
        points_pre = np.random.rand(2, d)   # generate 2d points
        points_pre[points_pre <= 0.5] = -1
        points_pre[points_pre > 0.5] = 1
        cos_theta = np.dot(points_pre[0], points_pre[1])/(np.linalg.norm(points_pre[0])*np.linalg.norm(points_pre[1]))
        angle_list.append(round(math.degrees(math.acos(cos_theta)), 2))

    return angle_list


def pmf(angle, d):
    '''This function plots pmf'''
    ePMF = ECDF(angle)
    plt.plot(ePMF.x, ePMF.y)    # plot
    plt.xlabel('Angle')     # x-axis label
    plt.ylabel('ePMF')      # y-axis label
    plt.title('ePMF for angles for dimension d = {}'.format(d))     # title
    plt.show()


if __name__ == '__main__':

    # Calculations for dimension 10
    d = 10
    angle = cal_angle(d)                          # calculate angles between half diagonals for d = 10
    pmf(angle, d)                                  # Plot PMF
    print('Dimension 10 :\n', stats.describe(angle), '\n')  # display stats

    # Calculations for dimension 100
    d = 100
    angle = cal_angle(d)                          # calculate angles between half diagonals for d = 100
    pmf(angle, d)                                  # Plot PMF
    print('Dimension 100 :\n', stats.describe(angle), '\n')     # display stats

    # Calculations for dimension 1000
    d = 1000
    angle = cal_angle(d)                          # calculate angles between half diagonals for d = 1000
    pmf(angle, d)                                  # Plot PMF
    print('Dimension 1000 :\n', stats.describe(angle), '\n')    # display stats

