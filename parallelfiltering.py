# parallelfiltering.py: Code for synchronous parallel computations

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

import onlineldavb, batchvb, ep_lda, ep2_lda
#import wikirandom
import copy,math
from multiprocessing import Pool, cpu_count

def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def runBatchVB((W, K, docs, alpha, lam, maxiters,thr,hbb)):
        batchVB = batchvb.BatchLDA(W, K, docs, alpha, lam, useHBBBound = hbb)
        lam = batchVB.train(maxiters, thr)
        return lam

# For ParallelFilteringEP
def runBatchEP((W, K, docs, alpha, lam, maxiters, threshold, useNewton)):
        ep = ep_lda.EP_LDA(W, K, docs, alpha, lam, useNewton)
        lam = ep.train(maxiters, threshold)
        return lam

# For ParallelFilteringEP2
def runBatchEP2((W, K, docs, alpha, lam, maxiters, threshold, useNewton)):
        ep2 = ep2_lda.EP2_LDA(W, K, docs, alpha, lam, useNewton)
        lam = ep2.train(maxiters, threshold)
        return lam
    

class ParallelFiltering:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useHBB, numChunks, batchsize):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """
        self._str = "ParallelFiltering_%d_%r_%g_%d_%d" % (maxiters,useHBB,threshold,numChunks,batchsize)
        self._thresh = threshold
        self._maxiters = maxiters
        self._hbb = useHBB
        self._numChunks = numChunks
        
        self._pool = Pool(min(cpu_count(),numChunks))
        self._K = K
        self._W = W

        self._alpha = alpha
        if numpy.isscalar(eta):
           self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)
        #self._lambda = #numpy.random.gamma(100., 1./100., (self._K, self._W))

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        sizeOfChunks = int(math.ceil(len(docs) / float(self._numChunks)))
        chunks = chunk(docs, sizeOfChunks)
        estimates = self._pool.map(runBatchVB, [ (self._W, self._K, docs, self._alpha, self._lambda, self._maxiters, self._thresh, self._hbb) 
                                for docs in chunks]) # runBatchVB((self._vocab, self._K, docs, self._alpha, self._lambda))]
        self._lambda = sum(estimates) - (len(estimates) - 1)*(self._lambda)
        return (self._alpha,self._lambda)



# Parallel filtering with EP
class ParallelFilteringEP:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton, numChunks, batchsize):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        maxiters: Max number of iterations to allow to converge
        threshold: Threshold for convergence
        useNewton: Boolean, if true we use Newton's method for solving Dirichlet moment-matching, else use approximate MLE
        """
        self._str = "ParallelFilteringEP_%d_%g_%r_%dx_%d" % (maxiters,threshold,useNewton,numChunks,batchsize)        
        self._numChunks = numChunks        
        self._pool = Pool(min(cpu_count(), numChunks))
        self._K = K       
        self._W = W
        self._maxiters = maxiters
        self._threshold = threshold
        self._useNewton = useNewton

        self._alpha = alpha
        if numpy.isscalar(eta):
           self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        sizeOfChunks = int(math.ceil(len(docs) / float(self._numChunks)))
        chunks = chunk(docs, sizeOfChunks)
        estimates = self._pool.map(runBatchEP,
           [ (self._W, self._K, docs, self._alpha, self._lambda, self._maxiters, self._threshold, self._useNewton) for docs in chunks])
        self._lambda = sum(estimates) - (len(estimates) - 1) * self._lambda
        return (self._alpha, self._lambda)



# Parallel filtering with fake EP
class ParallelFilteringEP2:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton, numChunks, batchsize):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        maxiters: Max number of iterations to allow to converge
        threshold: Threshold for convergence
        useNewton: Boolean, if true we use Newton's method for solving Dirichlet moment-matching, else use approximate MLE
        """
        self._str = "ParallelFilteringEP2_%d_%g_%r_%dx_%d" % (maxiters,threshold,useNewton,numChunks,batchsize)        
        self._numChunks = numChunks
        self._pool = Pool(min(cpu_count(), numChunks))
        self._K = K       
        self._W = W
        self._maxiters = maxiters
        self._threshold = threshold
        self._useNewton = useNewton

        self._alpha = alpha
        if numpy.isscalar(eta):
           self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        sizeOfChunks = int(math.ceil(len(docs) / float(self._numChunks)))
        chunks = chunk(docs, sizeOfChunks)
        estimates = self._pool.map(runBatchEP2,
           [ (self._W, self._K, docs, self._alpha, self._lambda, self._maxiters, self._threshold, self._useNewton) for docs in chunks])
        self._lambda = sum(estimates) - (len(estimates) - 1) * self._lambda
        return (self._alpha, self._lambda)
