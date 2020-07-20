# Assignment 6.1
# Exercise 12.4.3 (a): knn with one neighbors
# Author: Saurabh Biswas
# DSC550 T302

# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_plot(x,y):
    """This function accepts two 1D arrays and plots a setp graph"""
    plt.step(x, y, where='mid')     # step graph
    plt.scatter(x, y)    # show data points on the plot
    plt.xticks(x)   # set current tick location
    plt.title('One nearest neighbor', fontsize=25)   # add title
    plt.xlabel('Query q', fontsize=20)   # x-axis label
    plt.ylabel('f(q)', fontsize=20)      # y-axis label
    plt.show()


if __name__ == '__main__':
    input_array = np.array([[1, 1], [2, 2], [4, 3], [8, 4], [16, 5], [32, 6]])    # input dataset
#    input_array = np.array([[1, 1], [2, 2], [3, 4], [4, 8], [5, 4], [6, 2], [7, 1]])    # input dataset
    df1 = pd.DataFrame(input_array)
    x = df1.iloc[:, 0]
    y = df1.iloc[:, 1]
    make_plot(x, y)
