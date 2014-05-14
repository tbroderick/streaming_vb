# utils.py: Utility functions

# This code suite is largely adapted from the online VB (aka stochastic
# variational Bayes) code of
# Matthew D. Hoffman, Copyright (C) 2010
# found here: http://www.cs.princeton.edu/~blei/downloads/onlineldavb.tar
# and also of 
# Chong Wang, Copyright (C) 2011
# found here: http://www.cs.cmu.edu/~chongw/software/onlinehdp.tar.gz
#
# Adapted by: Nick Boyd, Tamara Broderick, Andre Wibisono, Ashia C. Wilson
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import sys, re, time, string
import copy
import numpy as n
from scipy.special import psi, polygamma
from random import shuffle


# This is used in BatchVB's do_e_step
meanchangethresh = 0.00001

def unzipDocs(docs):
    wordids = [ids for (ids,cts) in docs]
    wordcts = [cts for (ids,cts) in docs]
    return (wordids,wordcts)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def strToBool(s):
    """
    Convert a string s into a Boolean value.
    """
    return (s.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh', 'indubitably'])

##############################################
# The following methods are called in EP_LDA #
##############################################

def unzipDocsShuffleWords(docs):
    """
    Extracts the (shuffled) list of wordids (i.e. the integer ids from the
    vocab list) for each doc in docs. The output format is a list of lists,
    one for each document. Note that the length may be different for each
    document.
    """
    D = len(docs)  # number of documents
    wordids = list()  # placeholder for output
    for d in range(0, D):
        (ids, cts) = docs[d]
        Nd = sum(cts)  # number of words in doc d
        # First populate wordids_d with the ordered ids,
        # e.g. if ids = [3,8,20] and cts = [1,3,2], then
        # wordids_d = [3,8,8,8,20,20]
        wordids_d = list()
        for i in range(0, len(ids)): 
            wordids_d.extend([ids[i]] * cts[i])
        # Shuffle wordids_d and insert to wordids
        shuffle(wordids_d)
        wordids.append(wordids_d)
    return wordids


def dirichlet_mle_newton(e_p, e_p2, e_logp, maxiters = 20, thr = 1e-4, silent = False):
    """
    Finds the MLE for the K-dimensional Dirichlet distribution from observed data,
    i.e. the solution alpha_1, ..., alpha_K > 0 to the moment-matching equations
        psi(alpha_k) - psi(sum(alpha)) = E[log p_k]
    where the expectation on the right hand side is with respect to the empirical
    distribution.

    Input: e_p, a vector of length K containing the empirical expectations E[p_k], i.e. e_p.ndim == 1 and len(e_p) == K
           e_p2, the empirical expectations E[p_k^2], the same format as e_p
           e_logp, the empirical expectations E[log p_k], the same format as e_p
           maxiters, the maximum number of Newton-Raphson iterations
           thr, the threshold for convergence 
    Output: alpha, a vector of length K containing the parameters alpha_1, ..., alpha_K

    This method uses the first and second empirical moments e_p and e_p2 to initialize
    the alpha values (by approximately matching the first and second moments), and then
    uses Newton-Raphson method to refine the estimates.

    This method is based on the first section of Minka's paper:
    http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf
    """

    # For initialization: First compute the approximate sum(alpha)
    alpha0 = (sum(e_p - e_p2)) / (sum(e_p2 - e_p ** 2))

    # Then compute the initial alpha
    alpha = alpha0 * e_p

    # Do Newton-Raphson iterations
    for iteration in range(0, maxiters):
        sum_alpha = sum(alpha)
        g = psi(alpha) - psi(sum_alpha) - e_logp
        z = polygamma(1, sum_alpha)  # polygamma(1,z) is the trigamma function psi_1(z)
        q = polygamma(1, alpha)
        b = sum(g / q) / (1 / z - sum(1 / q))
        alpha_new = alpha - (g + b) / q

        # this is a hack, but if some of alpha_new's components are negative, make them positive
        alpha_new[alpha_new < 0] = alpha / 5  # / 5 is arbitrary, as long as the end result is positive

        # Update alpha and check for convergence
        delta = max(abs(alpha - alpha_new))
        alpha = alpha_new
        if delta < thr:
            # cur_gap = psi(alpha) - psi(sum(alpha)) - e_logp
            # if not silent:
            #     print "Dirichlet-MLE-Newton converged in " + str(iteration) + " iterations, gap = " + str(cur_gap)
            break
        if iteration >= maxiters - 1:
            cur_gap = psi(alpha) - psi(sum(alpha)) - e_logp
            if not silent:
                print "Dirichlet-MLE-Newton did not converge after " + str(iteration) + " iterations, gap = " + str(cur_gap)
    return alpha
