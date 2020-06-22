# Assignment 2.1
# Exercise 3.2.1
# Finding Shingles
# Author: Saurabh Biswas
# DSC550 T302

# import regex library
import re

sentence_3_2 = "The most effective way to represent documents as sets, for the purpose of iden-tifying" \
               " lexically similar documents is to construct from the document the set of short strings" \
               " that appear within it."

# We wil take out all special character, We will also replace multiple blanks with a single blank.

new_sentence = re.sub('[^a-zA-Z0-9 \n]', '', sentence_3_2)  # take out all special char
new_sentence = re.sub(' +', ' ', new_sentence)    # replace multiple blank by a single blank

print("First ten 3-shingles are:")

for x in range(0, 10):  # run loop for 10 times
    print(new_sentence[x:x+3])  # print 3-shingles value


