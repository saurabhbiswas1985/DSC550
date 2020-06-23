# Assignment 2.1
# Nonderivable Itemset
# Author: Saurabh Biswas
# DSC550 T302

#newArr = x[x[:,0]=='1 ']
#print(newArr[0,1])

import pandas as pd
from itertools import combinations


def read_file():
    """
    Reads file and returns dictionary
    """
    df1 = pd.read_csv('itemsets.txt', sep = "-", header=None)     # read file

    sup_dict = {}   # empty support dictionary
    for i, sets in enumerate(df1[0]):  # run loop to get the iterable object
        sets = sets.rstrip()    # strip extra space from right
        sets = sets.replace(' ', ',')   # replace blank with comma
        sup_dict[sets] = df1[1][i]  # add it into dictionary

    ndi_dict = {}   # Empty dict
    df2 = pd.read_csv('ndi.txt', header=None)   # read file
    df2 = df2.apply(lambda x: x.str.replace(' ',','))   # replace in between spaces with comma

    for i, sets in enumerate(df2[0]):   # run loop to get the iterable object
        ndi_dict[i] = sets  # add it into dict

    return sup_dict, ndi_dict


def get_pwrset(ndi, ln):
    '''returns all combination of subsets from a given set'''
    pwr_set = []    # empty list for power set combination
    print('ln=', ln)
#    for sets in combinations(ndi, ln):  # run a loop to generate power set
#        print(sets)
#        pwr_set.append(sets)
    print('dict=', ndi[0])

    for i in range(1 << ln):
        pwr_set.append([ndi[j] for j in range(ln) if (i & (1 << j))])
    print(pwr_set)
    return pwr_set


def get_boundary(subset, ndi1, sup_dict1):
    """ This function retrieves support value from support dictionary and calculate
        boundary value"""

    boundary = 0.0
    for i in range((len(ndi1) - 1), 0, -1):    # Iterate from higher combination to lower combination
        pwr_set1 = get_pwrset(ndi1, i)          # Get power set of set for a length
        for pwr_sub in pwr_set1:                  # Check all subsets in powers et
            itm = all(item in pwr_sub for item in subset)   # Get all items
            if itm or subset == ():
                b = int(sup_dict1[pwr_sub]) * (-1) ** (len(ndi1)+1-i)   # boundary calculation formula
                boundary += b

    if subset == ():                            # in case subset is empty
        null = 0
        for dict_value in sup_dict1.values():
            null += int(dict_value)
        boundary += null * (-1.0) ** (len(ndi1) + 1)

    return boundary


def derivable_func(sup_dict, ndi_dict):
    """
    This function checks derivability using passed support dictionary and ndi dict
    """
    outcome_list = []   # empty list
    for ndi in ndi_dict.values():   # loop on given ndi dictionary
        upper_bound = []
        lower_bound = []
        for ln in range(1,len(ndi)+1):  # iterate over length of each ndi element
            temp_list = []
            temp_list.append(ndi)   # create a ndi list
#            print(temp_list)
            pwr_set = get_pwrset(temp_list, ln)                  # Get subsets of sets for each length
#            print(pwr_set)
            len_diff = len(temp_list) - len(pwr_set[0])          # Calculate the difference in length
            for subset in pwr_set:                           # Check each subset of powerset of set
                if len_diff % 2 == 0:                      # Is difference Even ? Yes, lower bound
                    lower_bound.append(get_boundary(subset, temp_list, sup_dict))   # calculate bound and append
                else:                                      # Otherwise Odd, upper bound
                    upper_bound.append(get_boundary(subset, temp_list, sup_dict))    # calculate bound and append

        # If maximum value of lower boundary and minimum value upper boundary then the
        # itemset is derivable from its subsets.

        # calculate max lower boundary
        if max(lower_bound) < 0:
            lower_bound = 0
        else:
            lower_bound = max(lower_bound)

        # calculate min upper boundary
        upper_bound = min(upper_bound)

        # Check if they are equal
        if lower_bound == upper_bound:
            item_superset = 'derivable'
        else:
            item_superset = 'non-derivable'

        # Add result to the list
        outcome_list.append((ndi,':', [lower_bound, upper_bound], item_superset))

    df3 = pd.DataFrame(outcome_list)     # Change it to dataframe
    df3.to_csv('ndi_result.txt')    # write output

    return


if __name__ == '__main__':

    sup_dict, ndi_dict = read_file()

    derivable_func(sup_dict, ndi_dict)  # invoke function to check NDI
