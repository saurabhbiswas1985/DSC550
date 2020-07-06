# Assignment 4.1
# Calculate Mahalanobis Distance - Exercise 7.3.5
# Autazhor: Saurabh Biswas
# DSC550 T302

# Import important libraries
import numpy as np


def calculate_mahalanobis_distance(point, centroid, sd, dimension):
    """This function take a point in, centroid and dimension as an input
        and return mahalanobis distance"""
    x = 0   # initialize
    for i in range(dimension):
        x += ((point[i]-centroid[i]) / sd[i]) ** 2  # calculate sum of squares

    return x ** 0.5     # return mahalanobis distance


if __name__ == '__main__':

    dimension = 3
    sd = np.array([2, 3, 5])    # standard deviation
    centroid = np.array([0, 0, 0])  # origin point
    point = np.array([1, -3, 4])    # given point

    mahalanobis_dist = calculate_mahalanobis_distance(point, centroid, sd, dimension)

    print('The Mahalanobis distance between (0,0,0) and (1,-3,4) is: {:.2f}'.format(mahalanobis_dist))


