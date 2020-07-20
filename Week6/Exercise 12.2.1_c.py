# Assignment 6.1
# Exercise 12.2.1 (c): Perceptrons basic method with variable threshold
# Author: Saurabh Biswas
# DSC550 T302

import numpy as np


def basic_perceptron(input_array, learn_rate):
    """This function accepts an input array and learning rate.
        Then it perform basic perceptron algorithm with variable
        threshold and display data """

    w_vector = [0, 0, 0, 0, 0, 0]  # weight vector

    for itr_cnt in range(25):     # it will try to get to convergence for 25 times
        convergence_switch = 'y'    # initialize to converged
        for i in range(len(input_array)):
            w_dot_x = np.dot(w_vector, input_array[i][0:6])     # get dot product w_vector and each array row

            if w_dot_x == 0:
                w_vector += learn_rate * input_array[i][0:6] * input_array[i][-1]   # w = w+nyx
                convergence_switch = 'n'    # set convergence switch to No
            elif np.sign(w_dot_x) != np.sign(input_array[i][-1]):
                w_vector += learn_rate * input_array[i][0:6] * input_array[i][-1]  # w = w+nyx
                convergence_switch = 'n'  # set convergence switch to No
            else:
                continue
        if convergence_switch == 'y':   # if converged then exit
            break

    if convergence_switch == 'y':
        print('The dataset has converged  after {} iteration\n'.format(itr_cnt))
        print('Converged values of weight vector:', w_vector)
    else:
        print("The dataset hasn't converged")


if __name__ == '__main__':
    # data from 12.6 after changing the row 2, col 5 to 1 (nigeria).
    # Added column 6 with constant -1 for theta.
    input_data = np.array([[1, 1, 0, 1, 1, -1, +1], [0, 0, 1, 1, 1, -1, -1], [0, 1, 1, 0, 0, -1, +1],
                           [1, 0, 0, 1, 0, -1, -1], [1, 0, 1, 0, 1, -1, +1], [1, 0, 1, 1, 0, -1, -1]])

    lrt = 0.5    # assume learning rate as 0.5

    basic_perceptron(input_data, lrt)    # invoke perceptron function
