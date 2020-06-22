# Assignment 2.1
# Exercise 3.3.3
# Minhash Calculation
# Author: Saurabh Biswas
# DSC550 T302

import numpy as np
from sklearn.metrics import jaccard_score


def hsh(x):
    '''
    hash function.
    :param x: Element number
    :return: list of hash values
    '''

    list_x = []
    y = (2 * x + 1) % 6
    list_x.append(y)
    y = (3 * x + 2) % 6
    list_x.append(y)
    y = (5 * x + 2) % 6
    list_x.append(y)
    return list_x


def minhash(data, hash_num):
    '''
    :param data: Matrix
    :param hash_num: number of hash functions
    :return: minhash signature
    '''

    rows = len(data)    # row count of the passed matrix
    cols = len(data[0]) # column count of the passed matrix

    sig_hash = []   # an empty list for sig_hash matrix
    hash_list = []

    # run a loop to assign infinity value to each row/col in sig_hash matrix
    for i in range(hash_num):
        list_row = []

        for j in range(cols):
            list_row.append(float('inf'))

        sig_hash.append(list_row)

    # run a loop to access each element of the matrix
    for r in range(rows):
        hash_row = hsh(r)   # invoke the function to get the hash values
        hash_list.append(hash_row)  # append to the hash function value list

        for c in range(cols):
            # if element is 0 then skip to next element in the same row
            if data[r][c] != 0:
                for i in range(hash_num):   # run a loop to examine each hash function value
                    if sig_hash[i][c] > hash_row[i]:    # if sig(i,c) > h(i) then replace
                        sig_hash[i][c] = hash_row[i]    # sig(i,c) with h(i)

    return sig_hash, hash_list


def list_order(list_a):
    ''' This function accepts an list and verify whether any one of its column is in order'''
    row = len(list_a)  # get the row count
    col = len(list_a[0])  # get column count

    array_unsorted = np.array(list_a)   # convert into an array
    array_sorted = np.sort(array_unsorted, axis=0)  # sort based on column

    for c in range(col):    # we want to check each which has func is in order from 0 - 5.
        for r in range(row):
            if array_sorted[r, c] != r:
                break
            elif array_sorted[r, c] == row-1:
                print('hash func {} is a true permutation'.format(c+1))
        else:
            continue


def Jaccard_Similarites(list1, list2):
    '''
    This function takes two list as input and calculates Jaccard Similarities
    between them.
    :param list1: set1
    :param list2: set2
    :return: Jaccard Similarity Value
    '''
    set1 = set(list1)   # convert list1 into a set
    set2 = set(list2)   # convert list2 into a set
    union1 = set.union(set1, set2)    # get union of set1 and set2
    intersection1 = set.intersection(set1, set2)  # get intersection of set1 and set2
    jac_sim_value = len(intersection1) / len(union1)
    return jac_sim_value   # return results


if __name__ == '__main__':

    data = [[0,1,0,1],[0,1,0,0],[1,0,0,1],[0,0,1,0],[0,0,1,1],[1,0,0,0]]    # data matrix
    x, y = minhash(data, 3)    # invoke minhash function
    print('Minhash Signature:', x)    # print minhash singnature
    print('Hash Function values:', y)

    array = np.array(x)

    list1 = array[:, 0]
    list2 = array[:, 1]
    list3 = array[:, 2]
    list4 = array[:, 3]

    js_12 = Jaccard_Similarites(list1, list2)   # invoke the function
    print(' Estimated Jaccard similarities between set 1 and 2 is: {:.2f}'.format(js_12))

    js_13 = Jaccard_Similarites(list1, list3)   # invoke the function
    print(' Estimated Jaccard similarities between set 1 and 3 is: {:.2f}'.format(js_13))

    js_14 = Jaccard_Similarites(list1, list4)   # invoke the function
    print(' Estimated Jaccard similarities between set 1 and 4 is: {:.2f}'.format(js_14))

    js_23 = Jaccard_Similarites(list2, list3)   # invoke the function
    print(' Estimated Jaccard similarities between set 2 and 3 is: {:.2f}'.format(js_23))

    js_24 = Jaccard_Similarites(list2, list4)   # invoke the function
    print(' Estimated Jaccard similarities between set 2 and 4 is: {:.2f}'.format(js_24))

    js_34 = Jaccard_Similarites(list3, list4)   # invoke the function
    print(' Estimated Jaccard similarities between set 3 and 4 is: {:.2f}'.format(js_34))

    # actual Jaccard similarities
    array = np.array(data)
    list1 = array[:, 0]
    list2 = array[:, 1]
    list3 = array[:, 2]
    list4 = array[:, 3]

    js_12 = jaccard_score(list1, list2)  # invoke the function
    print(' Actual Jaccard similarities between set 1 and 2 is: {:.2f}'.format(js_12))

    js_13 = jaccard_score(list1, list3)  # invoke the function
    print(' Actual Jaccard similarities between set 1 and 3 is: {:.2f}'.format(js_13))

    js_14 = jaccard_score(list1, list4)  # invoke the function
    print(' Actual Jaccard similarities between set 1 and 4 is: {:.2f}'.format(js_14))

    js_23 = jaccard_score(list2, list3)  # invoke the function
    print(' Actual Jaccard similarities between set 2 and 3 is: {:.2f}'.format(js_23))

    js_24 = jaccard_score(list2, list4)  # invoke the function
    print(' Actual Jaccard similarities between set 2 and 4 is: {:.2f}'.format(js_24))

    js_34 = jaccard_score(list3, list4)  # invoke the function
    print(' Actual Jaccard similarities between set 3 and 4 is: {:.2f}'.format(js_34))

    list_order(y)
