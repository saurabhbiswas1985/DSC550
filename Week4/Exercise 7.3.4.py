# Assignment 4.1
# BFR Algorithm - Exercise 7.3.4
# Autazhor: Saurabh Biswas
# DSC550 T302

# import required libraries
import numpy as np
import math


def bfr_calculation(c):
    """ This function takes a cluster as an input and calculates
        N, SUM, SUMSQ, variance and standard deviation"""
    n = len(c)  # element count of the cluster

    sum_c = np.sum(c, axis=0)   # calculate the vector sum
    sumsq_c = np.sum(np.square(c), axis=0)  # calculate the sum of squares
    variance_c = np.var(c, axis=0, dtype=np.float32)    # calculate variance
    sd_c = np.std(c, axis=0, dtype=np.float32)    # calculate standard deviation

    print('Cluster - \n', c)
    print('N for this cluster is     : ', n)
    print('SUM for this cluster is   : ', sum_c)
    print('SUMSQ for this cluster is : ', sumsq_c)
    print('Variance for this cluster is :', variance_c)
    print('Standard Deviation for this cluster is :', sd_c, '\n')

    return


if __name__ == '__main__':

    c1 = np.array([[4, 8], [6, 8], [4, 10], [7, 10]])
    c2 = np.array([[9, 3], [12, 3], [11, 4], [10, 5], [12, 6]])
    c3 = np.array([[2, 2], [5, 2], [3, 4]])

    # invoke BFR calculation function
    bfr_calculation(c1)
    bfr_calculation(c2)
    bfr_calculation(c3)



