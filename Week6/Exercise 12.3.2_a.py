# Assignment 6.1
# Exercise 12.3.2_a: Support Vector
# Author: Saurabh Biswas
# DSC550 T302

# import required libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC


def svc_model(df1):
    """This function build and trains SVC models and returns support vectors"""
    clf = SVC()     # invoke sklearn svc model
    clf.fit(df1.iloc[:, 0:3], df1.iloc[:, -1])    # train the model
    print('Number of support vector for each class: ', clf.n_support_)
    print('The support vectors are \n', clf.support_vectors_)


if __name__ == '__main__':
    dataset = np.array([[3, 4, 5, 1], [2, 7, 2, 1], [5, 5, 5, 1],
                        [1, 2, 3, -1], [3, 3, 2, -1], [2, 4, 1, -1]])

    df = pd.DataFrame(dataset)     # convert into pandas dataframe
    svc_model(df)  # invoke the function

    print('All are support vectors from this training dataset')
