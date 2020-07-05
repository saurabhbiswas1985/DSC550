# Assignment 4.1
# Expectation Maximization Algorithm
# Autazhor: Saurabh Biswas
# DSC550 T302

# import required libraries
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
import sys
import time


def file_read(filename):
    """ This routine reads file into a dataframe"""
    df1 = pd.read_csv(filename, header=None)
    return df1


def em_init(df2,k):
    """This function takes a dataframe and number of cluster as input.
        It produces mu, sigma & pi"""
    n, d = df2.shape

    #   generate mu k X d and pi
    mus_list = []
    pis_list = []
    for i in range(k):
        for j in range(d):
            mu = np.random.uniform(df2[j].min(), df2[j].max())
            mus_list.append(mu)
        pis_list.append(1/k)

    mus = np.array(mus_list)
    mus = mus.reshape(k, d)
    pis = np.array(pis_list)

    #   generate sigma d X d
    sigmas = np.array([np.eye(d)] * d)

    return mus, sigmas, pis


def em_algorithm(df2,k,mus,sigmas,pis,eps,maxitr):
    """ This is the implementation of EM algorithm. it consists
        of two steps - Expectation and Maximization. It repeats
        (ll_new-ll_old) becomes less than epsilon value or reaches
        the maximum number of iterations"""
    data = df2.to_numpy()     # convert to numpy array
    n, d = data.shape    # get row and attribute count
    ll_old = 0

    for iteration in range(maxitr):
        ll_new = 0

        # Expectation-step
        ws = np.zeros((k, n))
        for i in range(k):
            for j in range(n):
                # posterior probability
                ws[i, j] = pis[i] * mvn(mus[i], sigmas[i]).pdf(data[j])
        ws /= ws.sum(0)

        # Maximization-step

        # re-estimate mus
        mus = np.zeros((k, d))
        for i in range(k):
            for j in range(n):
                mus[i] += ws[i, j] * data[j]
            mus[i] /= ws[i, :].sum()

        #   re-estimate sigmas
        sigmas = np.zeros((k, d, d))
        for i in range(k):
            for j in range(n):
                ys = np.reshape(data[j] - mus[i], (d, 1))
                sigmas[i] += ws[i, j] * np.dot(ys, ys.T)
            sigmas[i] /= ws[i, :].sum()

        # re-estimate pis
        pis = np.zeros(k)
        for i in range(k):
            for j in range(n):
                pis[i] += ws[i, j]
        pis /= n

        # convergence test
        ll_new = 0.0
        for i in range(k):
            s = 0
            for j in range(n):
                s += pis[i] * mvn(mus[i], sigmas[i]).pdf(data[j])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < eps:   # if less than epsilon then stop iteration
            break

        ll_old = ll_new

    return ll_new, mus, sigmas, pis, iteration, ws


if __name__ == '__main__':
    start = time.time()

    # Check if the command line arguments are given
    if len(sys.argv) < 4:
        print('no/less arguments passed')
        sys.exit()

    print('Filename: ', sys.argv[1])
    print('Cluster Value: ', sys.argv[2])
    print('Epsilon Value: ', sys.argv[3])

    filename = (sys.argv[1])

    try:
        k = int(sys.argv[2])
    except ValueError:
        print('Enter a numeric integer')
        sys.exit()

    try:
        eps = float(sys.argv[3])
    except ValueError:
        print('Enter a float value')
        sys.exit()

    # read and prepare data
    df1 = file_read(filename)
    df2 = df1.iloc[:, 0:4]  # use only first four column

    # initialize
    mus, sigmas, pis = em_init(df2,k)

    # invoke EM algorithm
    maxitr = 10000  # maximum number of iteration
    ll_new, mus, sigmas, pis, iteration, ws = em_algorithm(df2, k, mus, sigmas, pis, eps, maxitr)

    for c, mu in enumerate(mus):
        print(f'\nThe means for cluster {c} is {mu}')

    for c, si in enumerate(sigmas):
        print(f'\nThe covariance matrix for cluster {c} is:\n{si}')

    print(f'\nEM algorithm took {iteration} iterations to complete the convergence.')

    df3 = pd.DataFrame(ws)
    maxValueIndex = df3.idxmax(axis=0)
    print('The final cluster is:\n', maxValueIndex)
    df3 = pd.DataFrame(maxValueIndex)

    unique_elements, counts_elements = np.unique(maxValueIndex, return_counts=True)
    df4 = pd.DataFrame(counts_elements)
    print('Final size of each cluster:\n', df4)

    # concatenate cluster with original datarame
    df5 = pd.concat([df1, df3], axis=1, ignore_index=True)
    df5.columns = ['attr1', 'attr2', 'attr3', 'attr4', 'type', 'cluster']   # assign column name
    df5 = df5.assign(New=1)     # add a new column with constant value
    df5 = df5.groupby(['type', 'cluster'], as_index=False)['New'].sum()

    # get max count from confusion matrix
    cnt_1 = df5[df5.type == 'Iris-setosa'].New.max()
    cnt_2 = df5[df5.type == 'Iris-versicolor'].New.max()
    cnt_3 = df5[df5.type == 'Iris-virginica'].New.max()
    purity = (cnt_1+cnt_2+cnt_3) / df5.New.sum()
    print('\n The purity value is:{:.2f}'.format(purity))
