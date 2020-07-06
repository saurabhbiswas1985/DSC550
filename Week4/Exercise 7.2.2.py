# Assignment 4.1
# Hierarchical Clustering - Exercise 7.2.2
# Autazhor: Saurabh Biswas
# DSC550 T302

# Import important libraries
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import sys


def plot_dendrogram(model, points, plot_title):
    """ This function takes plot title, model and data point as input.
        And plots dendrogram"""

    model = model.fit(points)   # fit the hierarchical clustering
    children = model.children_  # The children of each non-leaf node

    # No Distance attribute for the object. But this is required for linkage_matrix.
    # Create uniform array of distance from the children 0 to 10.
    distance = np.arange(children.shape[0])

    count = np.arange(2, children.shape[0]+2)   # number of observation points 2 to 12

    try:
        linkage_matrix = np.column_stack([children, distance, count]).astype(float)
    except ValueError:
        print('Something is wrong!!! Try again')
        sys.exit()

    dendrogram(linkage_matrix)  # plot dendrogram
    plt.title(plot_title)   # add a title
    plt.show()      # show the plot

    return


if __name__ == '__main__':

    # Dataset given in example 7.2
    points = np.array([[2, 2], [5, 2], [9, 3], [12, 3],
                      [3, 4], [11, 4], [10, 5], [12, 6],
                      [4, 8], [6, 8], [4, 10], [7, 10]])

    model = AgglomerativeClustering(affinity='euclidean', linkage='ward')    # model with minimum distance
    plot_title = 'Hierarchical Clustering Tree for Minimum Distance'
    plot_dendrogram(model, points, plot_title)  # invoke the function to plot dendrogram

    model = AgglomerativeClustering(affinity='euclidean', linkage='average')    # model with avg distance
    plot_title = 'Hierarchical Clustering Tree for Avg Distance'
    plot_dendrogram(model, points, plot_title)  # invoke the function to plot dendrogram
