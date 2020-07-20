# Assignment 6.1
# Exercise 12.2.1 (d): Perceptrons winnow method with variable threshold
# Author: Saurabh Biswas
# DSC550 T302

import numpy as np


def weight_adjustment(w, x, adjustment_type):
    """This function does the winnow weight adjustment"""

    j = 0
    # run a loop to select positions from x where the value is 1.
    # multiply or divide corresponding position value by 2 in vector w
    for element in x:
        if element == 1:
            if adjustment_type == 'multiply':
                w[j] *= 2
            else:
                w[j] /= 2
        j += 1  # increment the index by 1

    return w    # return updated vector


def winnow_perceptron(input_array):
    """This function accepts an input array.
        Then it performs winnow perceptron algorithm with variable
        threshold and display data """

    w_vector = [1, 1, 1, 1, 1, 1]  # weight vector
    mis_class_old = 0  # misclassified training point count old
    mis_class_new = 0  # misclassified training point count old

    for itr_cnt in range(25):     # it will try to get to convergence for 25 times
        convergence_switch = 'y'    # initialize to converged
        for i in range(len(input_array)):
            w_dot_x = np.dot(w_vector, input_array[i][0:6])     # get dot product w_vector and each array row

            if w_dot_x > 0 and input_array[i][-1] == -1:
                w_vector[-1] *= 2   # update theta
                w_vector = weight_adjustment(w_vector, input_array[i][0:6], 'divide')
                mis_class_new += 1  # update misclassified count
                convergence_switch = 'n'  # set convergence switch to No
            elif w_dot_x < 0 and input_array[i][-1] == 1:
                w_vector[-1] /= 2   # update theta
                w_vector = weight_adjustment(w_vector, input_array[i][0:6], 'multiply')
                mis_class_new += 1  # update misclassified count
                convergence_switch = 'n'  # set convergence switch to No
            elif w_dot_x == 0:
                w_vector[-1] /= 2   # update theta
                w_vector = weight_adjustment(w_vector, input_array[i][0:6], 'multiply')
                mis_class_new += 1  # update misclassified count
                convergence_switch = 'n'  # set convergence switch to No
            else:
                continue

        # convergence criteria - max 25 iteration or no change in misclassified points count
        # or dataset converged.
        if convergence_switch == 'y':   # if converged then exit
            break
        elif mis_class_new == mis_class_old:    # no change in misclassified count
            break
        else:
            mis_class_old = mis_class_new  # store misclassified count
            mis_class_new = 0   # reset new counter to 0

    if convergence_switch == 'y':
        print('The dataset has converged  after {} iteration\n'.format(itr_cnt))
        print('Converged values of weight vector:', w_vector)
    else:
        print("The dataset hasn't converged")
        print('It has completed {} iteration\n'.format(itr_cnt+1))
        print('The number of misclassified points: {}\n'.format(mis_class_new))
        print('Values of weight vector:', w_vector)


if __name__ == '__main__':
    # data from 12.6 after changing the row 2, col 5 to 1 (nigeria).
    # Added column 6 with constant -1 for theta.
    input_data = np.array([[1, 1, 0, 1, 1, -1, +1], [0, 0, 1, 1, 1, -1, -1], [0, 1, 1, 0, 0, -1, +1],
                           [1, 0, 0, 1, 0, -1, -1], [1, 0, 1, 0, 1, -1, +1], [1, 0, 1, 1, 0, -1, -1]])

    winnow_perceptron(input_data)    # invoke perceptron function
