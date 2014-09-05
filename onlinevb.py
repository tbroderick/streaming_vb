# onlinevb.py

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

import onlineldavb, batchvb
#import wikirandom
import copy,math
from multiprocessing import Pool
def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
def getSuffStats((vocab, K, docs, alpha, lam)):
        batchVB = batchvb.BatchLDA(vocab, K, docs, alpha, lam)
        batchVB.set_lambda(copy.deepcopy(lam))
        return batchVB.do_e_step()
    

class OnlineVB:
    def __init__(self, vocab, K, alpha, eta, numThreads = 8):
        """
        Arguments:
        K: Number of topics
        vocab: A DICTIONARY of words to ids
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """
        self._numThreads = numThreads
        
        
        #self._pool = Pool(numThreads)
        self._vocab = vocab
        self._K = K
       
        self._W = len(self._vocab)

        self._alpha = alpha
        if numpy.isscalar(eta):
           self._lambda = copy.deepcopy(eta) * numpy.ones((self._K, self._W))
        else:
            self._lambda = copy.deepcopy(eta)
        self._updatect = 0

        self._Elogbeta = batchvb.dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)

    def updateEstimate(self,docs):
        #sizeOfChunks = int(math.ceil(len(docs)/float(self._numThreads)))
        #chunks = chunk(docs, sizeOfChunks)
        estimates = [getSuffStats((self._vocab, self._K, docs, self._alpha, self._lambda))]#self._pool.map(runBatchVB, [ (self._vocab, self._K, docs, self._alpha, self._lambda) for docs in chunks])
        self._lambda = self._lambda + estimates[0] 
        return self._lambda


