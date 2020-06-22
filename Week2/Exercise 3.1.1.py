# Assignment 2.1
# Exercise 3.1.1
# Jaccard Similarities
# Author: Saurabh Biswas
# DSC550 T302


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


# invoke main program
if __name__ == '__main__':
    # define three lists
    list1 = [1,2,3,4]
    list2 = [2,3,5,7]
    list3 = [2,4,6]

    js_12 = Jaccard_Similarites(list1, list2)   # invoke the function
    print('Jaccard similarities between set 1 and 2 is: {:.2f}'.format(js_12))

    js_23 = Jaccard_Similarites(list2, list3)   # invoke the function
    print('Jaccard similarities between set 2 and 3 is: {:.2f}'.format(js_23))

    js_13 = Jaccard_Similarites(list1, list3)  # invoke the function
    print('Jaccard similarities between set 1 and 3 is: {:.2f}'.format(js_13))
