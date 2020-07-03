# Assignment 4.1
# Clustering
# Autazhor: Saurabh Biswas
# DSC550 T302

# import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def load_data():
    """This function loads data from sklearn"""
    data = load_iris(as_frame=True)
    df1 = data['data']
    return df1


def replace_nan(old_centroid, new_centriod):
    """ This function takes old & new numpy array as an input and replaces any entire row
     with NaN in new array with corresponding old array row"""
    nrows = len(new_centriod)

    for r in range(nrows):  # run a loop to find out rows having all NaN and replace that with old row
        if all(np.isnan(new_centriod[r])):
            new_centriod[r] = old_centroid[r]
    return new_centriod


def distance(pt1, pt2):
    """ This function calculates the distance between two vectors"""
    return sum((i-j)*(i-j) for i, j in zip(pt1, pt2))


def centroid_distance(old_centroid, new_centroid):
    """This function calculates the distance between two centroids"""
    nrows = len(new_centroid)   # get the number of rows
    total_dist = 0
    for r in range(nrows):      # run a loop to calculate the distance between old and new centroids
        dist = distance(old_centroid[r], new_centroid[r])
        total_dist += dist
    return total_dist


def kmeans(df1, k_loop, eps):
    """ This function accepts dataframe, number of iteration for k value and epsilon
        and returns sum of squared deviation list and corresponding k value list"""

    n_row, n_col = df1.shape
    list_ssd = []
    k_value = []

    for k in range(1, k_loop+1):
        rand_indices = np.random.choice(n_row, size=k)  # get random indices
        centroids_df = df1.loc[rand_indices]    # set random centroids
        centroids = centroids_df.to_numpy()     # convert it into numpy array

        for i in range(1000):   # run upto 1000 iteration for each k-value
            distances_to_centroids = pairwise_distances(df1, centroids, metric='euclidean',squared=True)
            cluster_assignment = np.argmin(distances_to_centroids, axis=1)  # find minimum centroid distance
            new_centroids = np.array([df1[cluster_assignment == j].mean(axis=0) for j in range(k)])  # get new centroid
            new_centroids = replace_nan(centroids, new_centroids)   # invoke func to replace any null value

            old_cent_to_new_cent = centroid_distance(centroids, new_centroids)  # square of distance between old and new

            if old_cent_to_new_cent <= eps:     # if distance is less than epsilon then clustering is done
                break

            if np.all(centroids == new_centroids):  # if centroids are static then clustering is done
                break

            centroids = new_centroids   # assign new centroid and continue the loop

        ssd = 0     # sum of squared deviation

        for i in range(k):
            df2 = df1[cluster_assignment == i]
            # sum of squared distance of data points from corresponding centroids
            y = pairwise_distances(df2, [centroids[i]], metric='euclidean', squared=True)
            ssd += np.sum(y)
        list_ssd.append(ssd)
        k_value.append(k)

    return k_value, list_ssd


def plot_func(k_value, list_ssd, xlbl, ylbl, plot_title):
    """ This function plots a graph between k-value and SSD"""
    plt.plot(k_value, list_ssd)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(plot_title)
    plt.show()


def compute_gap(df1, k_loop, n_references):
    """ This function calculates gap statistics"""
    df1_array = df1.to_numpy()  # convert to numpy array
    if len(df1_array.shape) == 1:
        df1_array = df1_array.reshape(-1, 1)
    reference = np.random.rand(*df1_array.shape)    # generate random data
    df2 = pd.DataFrame(reference)   # convert to dataframe for invoking kmeans function
    ref_inertia = []    # reference inertia

    for _ in range(n_references):
        k_value, list_ssd = kmeans(df2, 10, 0.01)   # invoke kmeans with generated reference data
        ref_inertia.append(list_ssd)
    # convert to numpy array, find mean on column. This will provide mean reference inertia for all k values

    arr = np.array(ref_inertia)
    arr = np.mean(arr, axis=0)
    ref_inertia = arr.tolist()

    k_value, ondata_inertia = kmeans(df1, 10, 0.01)

    gap = np.log(ref_inertia) - np.log(ondata_inertia)

    return gap, np.log(ref_inertia), np.log(ondata_inertia), k_value


# invoke main program
if __name__ == '__main__':
    df1 = load_data()

    k_value, list_ssd = kmeans(df1, 10, 0.01)    # invoke k-means algorithm
    xlbl = 'k value'
    ylbl = 'SSD'
    plot_title = 'SSD v/s k - elbow plot'
    plot_func(k_value, list_ssd, xlbl, ylbl, plot_title)            # plot SSD vs k

    gap, ref_inertia, ondata_inertia, k_value = compute_gap(df1, 10, 500)
    xlbl = 'k value'
    ylbl = 'gap'
    plot_title = 'gap statistics plot'
    plot_func(k_value, gap, xlbl, ylbl, plot_title)     # plot gap v/s k

    plt.plot(k_value, ref_inertia,
             '-o', label='reference')
    plt.plot(k_value, ondata_inertia,
             '-o', label='data')
    plt.xlabel('k')
    plt.ylabel('log(inertia)')
    plt.show()

    print('Based on SSD-k plot, the optimum value of k is 3')
    print('Based on gap-k plot, the optimum value of k is 4,'
          'because after k=4, the graph is going downwards and'
          'we have maximum gap at 4')
    print('The optimum value of generated by SSD and gap are close.')
    print('gap statistics is a better method of choosing optimum'
          'value of k because SSD-k plot is not always providing'
          'the perfect elbow point.')