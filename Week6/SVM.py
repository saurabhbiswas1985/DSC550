# Assignment 6.1
# Problem 2: Support Vector Machines
# Author: Saurabh Biswas
# DSC550 T302

# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import joblib
from matplotlib.colors import ListedColormap


def file_read(filename, col_names):
    """This routine reads the file and loads it into a dataframe"""
    df1 = pd.read_csv(filename, sep=',', names=col_names)
    return df1


def get_scores(clf, x_train, x_test, y_train, y_test):
    """This routine fits the model and calculates the f1 score"""

    clf.fit(x_train, y_train)  # train the model
    y_pred = clf.predict(x_test)  # test the model
    return f1_score(y_test, y_pred)   # return the f1 score


def kfold_test(clf, x, y):
    """This function executes KFold test"""

    seed = 10  # set seed value

    # generate five fold cross validation sets
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    f1_scr_list = []    # list to hold f1 score for the test

    for train_index, test_index in kf.split(x):     # loop through cross validation set
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        f1 = get_scores(clf, x_train, x_test, y_train, y_test)
        f1_scr_list.append(f1)
#    print(sum(f1_scr_list))
    return sum(f1_scr_list) / len(f1_scr_list), clf  # mean f1 score & model


def svc_model(type, reg_parm, gama, tolerance):
    """This function takes a different parameter and builds a svc model """
    clf = SVC(kernel=type, C=reg_parm, gamma=gama, tol=tolerance)
    iteration_list = []  # empty list for each iteration
    iteration_list.append(type)
    iteration_list.append(reg_parm)
    iteration_list.append(gama)
    iteration_list.append(tolerance)
    return clf, iteration_list


def save_model(clf_tr, old_f1, new_f1):
    """ This function stores the optimum model"""

    model_file = 'finalized_model.sav'
    if old_f1 <= new_f1:
        # save the model to disk
        joblib.dump(clf_tr, model_file)
    return model_file


def build_svm(df1):
    """This function accepts a dataframe and builds svc model with
        different kernel and parameter"""

    result_list = []    # empty list for result
    x = df1.iloc[:, 0:2].to_numpy()  # select first two column as input attributes
    y = df1.iloc[:, -1].to_numpy()  # select last column as a target
    old_f1 = 0.0

    clf, iteration_list = svc_model('linear', 1.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit the model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)     # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('linear', 1000.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('linear', 1.0, 'scale', 1e-2)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit the model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('poly', 1.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('poly', 10.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('poly', 1.0, 50, 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('rbf', 1.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('rbf', 1.0, 'scale', 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model
    old_f1 = f1_mean

    clf, iteration_list = svc_model('rbf', 1.0, 50, 1e-3)
    f1_mean, clf_tr = kfold_test(clf, x, y)     # fit th model and get f1 score
    iteration_list.append(f1_mean)      # add mean f1 score into the list
    iteration_list.append(clf_tr)       # add trained model
    result_list.append(iteration_list)  # append result to the list
    model_filename = save_model(clf_tr, old_f1, f1_mean)  # save model

    col_list = ['Type', 'Regularization', 'gamma', 'tolerance', 'f1_mean', 'model']
    df2 = pd.DataFrame(result_list, columns=col_list)   # convert it into a datafarme
    return df2, model_filename


def versiontuple(v):
    """Returns a tuple"""
    return tuple(map(int, (v.split("."))))


# copied code as a reference to plot rbf kernel
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')     # markers for data points
    colors = ('blue', 'green', 'lightgreen', 'gray', 'cyan')    # plot colors
    cmap = ListedColormap(colors[:len(np.unique(y))])   # return a color map

    # plot the decision surface
    # get minimum and maximum range of both dimensions
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # get coordinate matrix from x1 & x2 dimensions
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.5, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


def plot_graph(df1, model_filename):
    """This function accepts a dataframe and svm model parameters and creates a plot"""

    X = df1.iloc[:, 0:2].to_numpy()  # select first two column as input attributes
    Y = df1.iloc[:, -1].to_numpy()  # select last column as a target
    loaded_model = joblib.load(model_filename)  # retrieve the model
    print('retrieved optimum model:', loaded_model)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
    plt.show()

    plot_decision_regions(X, Y, classifier=loaded_model)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    col_names = ['X1', 'X2', 'Target']  # column names
    df1 = file_read('iris-slwc.txt', col_names)

    print('Shape of the dataframe:', df1.shape, '\n')
    print(df1.head(5))

    df2, model_filename = build_svm(df1)    # invoke svm build with for different types and parameters
    print(df2.head(9))

    print('Best classifier is rbf kernel with regularization as 1.0, gamma as scale and'
          'tolerance as 0.001 ')

    plot_graph(df1, model_filename)


