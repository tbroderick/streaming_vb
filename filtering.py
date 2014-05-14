# filtering.py: Code for running streaming VB

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


import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb, batchvb, ep_lda, hdp, ep2_lda
#import wikirandom
import copy, math

    

class Filtering:
    def __init__(self, W, K, alpha, eta, maxiters, useHBBBound, threshold):
        """
        Arguments:
        K: Number of topics
        W: Total number of words in the vocab
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        maxiters: Max number of iterations to allow to converge
        useHBBBound: Use the strange, elbo-like bound from HBB
        threshold: Threshold for convergence
        """
        self._str = "Filtering_%d_%r_%g" % (maxiters,useHBBBound,threshold)
        self._K = K
        self._W = W
        self._useHBBBound = useHBBBound
        self._alpha = alpha
        self._maxiters = maxiters
        self._threshold = threshold
        #self._lambda = eta#1.0 #numpy.random.gamma(100., 1./100., (self._K, self._W))
        if numpy.isscalar(eta):
            self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)

        
    def update_lambda(self, docs):
        batchVB = batchvb.BatchLDA(self._W, self._K, docs, self._alpha, self._lambda, self._useHBBBound)
        self._lambda = batchVB.train(self._maxiters, self._threshold) #1E-4)
        return (self._alpha,self._lambda)

    def __str__(self):
        return self._str


# Filtering for HDP
class HDPFiltering:
    def __init__(self, W, eta, maxiters, threshold, T = 300, K = 30):
        """
        W: Total number of words in the vocab
        maxiters: max number of iterations to allow to converge
        """
        self._str = "HDPFiltering(%d,%g)" % (maxiters,threshold)
        self._W = W
        self._alpha = 1.0
        self._maxiters = maxiters
        self._threshold = threshold
        self._T = T
        self._K = K
        if numpy.isscalar(eta):
            self._lambda = eta * numpy.ones((T, self._W))
        else:
            self._lambda = copy.deepcopy(eta)
        self._gamma = numpy.ones((2, T-1))

    def __str__(self):
        return self._str
        
    def update_lambda(self, docs):

        batchhdp = hdp.hdp(self._T, self._K, self._W, self._lambda, self._gamma, self._alpha, docs)
        batchhdp.train(self._maxiters, self._threshold)
        self._lambda = batchhdp.m_beta
        self._gamma = batchhdp.m_var_sticks
        (alpha, lam) = batchhdp.hdp_to_lda() #batchVB.train(self._maxiters, self._threshold) #1E-4)
        return (alpha, lam)



# Filtering with EP as primitive
class FilteringEP:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton):
        """
        Arguments:
        K: Number of topics
        W: Total number of words in the vocab
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        maxiters: Max number of iterations to allow to converge
        threshold: Threshold for convergence
        useNewton: Boolean, if true we use Newton's method for solving Dirichlet moment-matching, else use approximate MLE
        """
        
        self._str = "FilteringEP_%d_%g_%r" % (maxiters,threshold,useNewton)
        self._K = K
        self._W = W
        self._alpha = alpha
        self._maxiters = maxiters
        self._threshold = threshold
        self._useNewton = useNewton
        if numpy.isscalar(eta):
            self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)
        
    def update_lambda(self, docs):
        ep = ep_lda.EP_LDA(self._W, self._K, docs, self._alpha, self._lambda, self._useNewton)
        self._lambda = ep.train(self._maxiters, self._threshold)
        return (self._alpha, self._lambda)



# Filtering with fake EP as primitive
class FilteringEP2:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton):
        """
        Arguments:
        K: Number of topics
        W: Total number of words in the vocab
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        maxiters: Max number of iterations to allow to converge
        threshold: Threshold for convergence
        useNewton: Boolean, if true we use Newton's method for solving Dirichlet moment-matching, else use approximate MLE
        """
        
        self._str = "FilteringEP2_%d_%g_%r" % (maxiters,threshold,useNewton)
        self._K = K
        self._W = W
        self._alpha = alpha
        self._maxiters = maxiters
        self._threshold = threshold
        self._useNewton = useNewton
        if numpy.isscalar(eta):
            self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)
        
    def update_lambda(self, docs):
        ep2 = ep2_lda.EP2_LDA(self._W, self._K, docs, self._alpha, self._lambda, self._useNewton)
        self._lambda = ep2.train(self._maxiters, self._threshold)
        return (self._alpha, self._lambda)
