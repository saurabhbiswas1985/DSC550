# Assignment 6.1
# Exercise 12.4.3 (a): knn with one neighbors
# Author: Saurabh Biswas
# DSC550 T302

# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_2nn_avg(input_points):
    """This function accepts input array points and calculates
    2 nearest neighbour average"""
    avg_array_list = []     # an empty list

    for i in range(len(input_points)-1):    # loop through the input
        x = (input_points[i]+input_points[i+1])/2   # get 2 points nearest neighbour avg\
        avg_array_list.append(x)    # add to the list
    return avg_array_list


def make_plot(x, y, x1, y1):
    """This function accepts two 1D arrays and plots a setp graph"""
    plt.step(x, y, where='mid')     # step graph
    plt.scatter(x1, y1)    # show data points on the plot
    plt.xticks(x1)   # set current tick location
    plt.title('Two nearest neighbor - Avg', fontsize=25)   # add title
    plt.xlabel('Query q', fontsize=20)   # x-axis label
    plt.ylabel('f(q)', fontsize=20)      # y-axis label
    plt.show()


if __name__ == '__main__':
    input_array = np.array([[1, 1], [2, 2], [4, 3], [8, 4], [16, 5], [32, 6]])  # input dataset

    df1 = pd.DataFrame(input_array)
    x1 = df1.iloc[:, 0]
    y1 = df1.iloc[:, 1]

    nn_array = get_2nn_avg(input_array)  # invoke function
    df1 = pd.DataFrame(nn_array)    # change it into dataframe
    x = df1.iloc[:, 0]
    y = df1.iloc[:, 1]

    make_plot(x, y, x1, y1)     # plot step graph and original points

