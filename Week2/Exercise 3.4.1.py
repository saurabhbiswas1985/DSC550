# Assignment 2.1
# Exercise 3.4.1
# S - Curve
# Author: Saurabh Biswas
# DSC550 T302


def s_curve(s, r, b):
    '''
    s-curve equation. accepts s,r & b and return s-curve value
    '''
    return 1 - (1-s ** r) ** b


def evaluate_s_curve(r, b):
    '''
    accepts r and b values and calculates s-curve value from 0.1 to 0.9 with an increment of 0.1
    '''
    for i in range(1, 10, 1):   # run a loop from 1 to 9 and then divide by 10 to get the fraction
        j = i/10
        x = s_curve(j, r, b)    # invoke s-curve equation
        print("S-curve value for s= {0}, r= {1} and b= {2}: {3:.4f}".format(j, r, b, x))
    print('\n')


# invoke main program
if __name__ == '__main__':
    evaluate_s_curve(r=3, b=10)     # invoke s-curve
    evaluate_s_curve(r=6, b=20)     # invoke s-curve
    evaluate_s_curve(r=5, b=50)     # invoke s-curve
