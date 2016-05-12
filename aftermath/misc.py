##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    ep = 6**0.5/(m+n)**0.5
    A0 = (random.rand(m,n) * 2 * ep) - ep

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0