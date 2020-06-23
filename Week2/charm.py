# Assignment 2.1
# Exercise 1
# Charm Algorithm
# Author: Saurabh Biswas
# DSC550 T302

# import required libraries
import pandas as pd
import sys
import time
from copy import copy


# Data Preparation Class
class DataPrep:
    transaction_list = []   # Create an empty list
    tid_count = 0  # Initializes tid count

    def read_file(self, filename):
        '''This functions reads data from a file and store them into a list'''
        with open(filename, 'r') as file:   # open input file
            tid = 1                       # Initialize TID
            for line in file:   # for each line from the file
                line = line.strip().split()
                for element in line:    # for each element from the line
                    self.transaction_list.append({'tid': tid, 'item': element})     # append data into list
                tid += 1    # keep on increasing the TID count
        self.tid_count = tid - 1    # get the actual TID count

    def data_convert(self):
        '''This function converts list into pandas DataFrame'''

        df = pd.DataFrame(self.transaction_list)    # convert into a data frame
        self.itemsGrouped = df.groupby(['item'])['tid'].apply(list)   # Group by elements
        self.itemsGrouped = pd.DataFrame({'item': self.itemsGrouped.index, 'tid': self.itemsGrouped.values})
        self.itemsGrouped['item'] = self.itemsGrouped['item'].apply(lambda x: {x})

    def get_frequent(self, minsup):
        ''' This function return groups having support greater than or equal to minsup '''
        return self.itemsGrouped[self.itemsGrouped['tid'].map(len) >= minsup]


class CharmAlgorithm:
    def __init__(self, minsup_value):
        ''' Initializes DataFrame output'''

        self.result = pd.DataFrame(columns=['item', 'tid', 'support'])
        self.minsup = minsup_value

    @staticmethod
    def replace_values(df, column, find, replace):
        """
        Static function to replace old values
        """
        for row in df.itertuples():
            if find <= row[column]:
                row[column].update(replace)

    def charm_criteria(self, row1, row2, items, new_item, new_tid):
        ''' Charm criteria '''
        if len(new_tid) >= self.minsup:
            if set(row1[2]) == set(row2[2]):                     # Property-1: same tid?
                items = items[items['item'] != row2[1]]          # remove row2
                find = copy(row1[1])                             # replace all 1st set with new_item
                self.replace_values(items, 1, find, new_item)
                self.replace_values(self.items_tmp, 1, find, new_item)
            elif set(row1[2]).issubset(set(row2[2])):            # Property-2: row1 tid subset of row2 tid
                find = copy(row1[1])                             # replace 1st with 2nd set of tid
                self.replace_values(items, 1, find, new_item)
                self.replace_values(self.items_tmp, 1, find, new_item)
            elif set(row2[2]).issubset(set(row1[2])):            # row2 tid is subset of row1 tid
                self.items_tmp = self.items_tmp.append({'item': new_item, 'tid': new_tid}, ignore_index=True)
            elif set(row1[2]) != set(row2[2]):                   # Property-3: if tids are unequal?
                # add {item, tid} to self.items_tmp
                self.items_tmp = self.items_tmp.append({'item': new_item, 'tid': new_tid}, ignore_index=True)


    def charm_apply(self, items_grouped):
        '''Charm algorithm'''
        # sort ascending support and reset the index
        s = items_grouped.tid.str.len().sort_values().index
        items_grouped = items_grouped.reindex(s).reset_index(drop=True)

        for r1 in items_grouped.itertuples():      # Apply CHARM Property for each row
            self.items_tmp = pd.DataFrame(columns=['item', 'tid'])  # Temp DataFrame to iterate over the results
            for r2 in items_grouped.itertuples():  # 2nd loop to compare with all other rows
                if r2[0] >= r1[0]:
                    item = set()
                    item.update(r1[1])
                    item.update(r2[1])
                    tid = list(set(r1[2]) & set(r2[2]))
                    self.charm_criteria(r1, r2, items_grouped, item, tid)   # Apply Charm criteria
            if not self.items_tmp.empty:         # is temp isn't empty reapply charm criteria
                self.charm_apply(self.items_tmp)
            # check if item subsumed
            is_subsumption = False
            for row in self.result.itertuples():
                if r1[1].issubset(row[1]) and set(row[2]) == set(r1[2]):
                    is_subsumption = True
                    break
            # append to result if element not subsumed
            if not is_subsumption:
                self.result = self.result.append({'item': r1[1], 'tid': r1[2], 'support': len(r1[2])},
                                                 ignore_index=True)
        return self.result


if __name__ == '__main__':
    start = time.time()

    # Check if the command line arguments are given
    if len(sys.argv) < 3:
        print('no arguments passed')
        sys.exit()

    print('Filename: ', sys.argv[1])
    print('Min Support Value: ', sys.argv[2])

    filename = (sys.argv[1])
    minsup = int(sys.argv[2])

    # Data Preparation
    data = DataPrep()       # invoke data preparation method
    data.read_file(filename)
    data.data_convert()     # Convert the data
    freq = data.get_frequent(minsup)    # Get the items > minimum support

    # Apply Charm Algorithm
    algorithm = CharmAlgorithm(minsup)
    df1 = algorithm.charm_apply(freq)
    print(df1)
