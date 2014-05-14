# asynchronous.py: Code for performing asynchronous computations

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

#filtering subroutine
from multiprocessing import Process, RawValue, Lock, RawArray, JoinableQueue
import numpy, batchvb, math, copy, ep_lda, ep2_lda
import ctypes as c
import multiprocessing as mp

def runBatchVB(workQueue, lockingPost):
    for (W, K, docs, alpha, maxiters,thr,hbb) in iter(workQueue.get, "none"):
        #read current parameter value (and copy)
        lam = copy.deepcopy(lockingPost.value())
        #calculate correction
        batchVB = batchvb.BatchLDA(W, K, docs, alpha, lam, useHBBBound = hbb)
        lam = batchVB.train( maxiters, thr)
        ss = (lam - batchVB._eta) 
        #apply correction
        lockingPost.increment(ss)
        workQueue.task_done()


def runBatchEP(workQueue, lockingPost):
    count = 0
    for (W, K, docs, alpha, maxiters, thr, useNewton) in iter(workQueue.get, "none"):
        #read current parameter value (and copy)
        lam = copy.deepcopy(lockingPost.value())
        #calculate correction
        ep = ep_lda.EP_LDA(W, K, docs, alpha, lam, useNewton)
        lam = ep.train(maxiters, thr)
        ss = (lam - ep._eta) 
        #apply correction
        lockingPost.increment(ss)
        workQueue.task_done()
        count += 1
        print "\tdone " + str(count)


def runBatchEP2(workQueue, lockingPost):
    count = 0
    for (W, K, docs, alpha, maxiters, thr, useNewton) in iter(workQueue.get, "none"):
        #read current parameter value (and copy)
        lam = copy.deepcopy(lockingPost.value())
        #calculate correction
        ep2 = ep2_lda.EP2_LDA(W, K, docs, alpha, lam, useNewton)
        lam = ep2.train(maxiters, thr)
        ss = (lam - ep2._eta) 
        #apply correction
        lockingPost.increment(ss)
        workQueue.task_done()
        count += 1
        print "\tdone " + str(count)


def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


class LockingPosterior(object):
    def __init__(self, eta):
        (self._K, self._W) = eta.shape
        self._lambda = RawArray('d', eta.reshape(self._K*self._W,))
        self._lock = Lock()
    def value(self):
        with self._lock:
            return numpy.frombuffer(self._lambda).reshape(self._K, self._W)
    def increment(self, ss):
        with self._lock:
            self._lambda[:] = (numpy.frombuffer(self._lambda).reshape(self._K,self._W) + ss).reshape(self._K * self._W, )



class ParallelFiltering:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useHBB, batchsize, numthreads):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """
        self._str = "Async_%d_%r_%g_%d_%d" % (maxiters,useHBB,threshold,batchsize,numthreads)
        self._thresh = threshold
        self._maxiters = maxiters
        self._hbb = useHBB
        self._batchsize = batchsize
        self._workQueue = JoinableQueue()


        self._K = K
        self._W = W

        self._alpha = alpha
        if numpy.isscalar(eta):
           etaA = eta * numpy.ones((self._K, self._W))
        else:
            etaA = eta
        self._posterior = LockingPosterior(etaA)
        
        #start workers
        self.workers = [Process(target=runBatchVB, args=(self._workQueue, self._posterior)) for i in range(numthreads)]
        for worker in self.workers:
            worker.start()

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        chunks = chunk(docs, self._batchsize)
        for doc_set in chunks:
            self._workQueue.put((self._W, self._K, doc_set, self._alpha, self._maxiters, self._thresh, self._hbb))
        self._workQueue.join()
        return (self._alpha, self._posterior.value())

    def shutdown(self):
        for worker in self.workers:
            worker.terminate()



class ParallelFilteringEP:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton, batchsize, numthreads):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """
        self._str = "AsyncEP_%d_%g_%r_%d_%d" % (maxiters, threshold, useNewton, batchsize, numthreads)
        self._thresh = threshold
        self._maxiters = maxiters
        self._useNewton = useNewton
        self._batchsize = batchsize
        self._workQueue = JoinableQueue()
        self._K = K
        self._W = W
        self._alpha = alpha
        if numpy.isscalar(eta):
           etaA = eta * numpy.ones((self._K, self._W))
        else:
            etaA = eta
        self._posterior = LockingPosterior(etaA)
        
        #start workers
        workers = [Process(target=runBatchEP, args=(self._workQueue, self._posterior)) for i in range(numthreads)]
        for worker in workers:
            worker.start()

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        chunks = chunk(docs, self._batchsize)
        for doc_set in chunks:
            self._workQueue.put((self._W, self._K, doc_set, self._alpha, self._maxiters, self._thresh, self._useNewton))
        self._workQueue.join()
        return (self._alpha, self._posterior.value())


class ParallelFilteringEP2:
    def __init__(self, W, K, alpha, eta, maxiters, threshold, useNewton, batchsize, numthreads):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        """
        self._str = "AsyncEP2_%d_%g_%r_%d_%d" % (maxiters, threshold, useNewton, batchsize, numthreads)
        self._thresh = threshold
        self._maxiters = maxiters
        self._useNewton = useNewton
        self._batchsize = batchsize
        self._workQueue = JoinableQueue()
        self._K = K
        self._W = W
        self._alpha = alpha
        if numpy.isscalar(eta):
           etaA = eta * numpy.ones((self._K, self._W))
        else:
            etaA = eta
        self._posterior = LockingPosterior(etaA)
        
        #start workers
        workers = [Process(target=runBatchEP2, args=(self._workQueue, self._posterior)) for i in range(numthreads)]
        for worker in workers:
            worker.start()

    def __str__(self):
        return self._str

    def update_lambda(self,docs):
        chunks = chunk(docs, self._batchsize)
        for doc_set in chunks:
            self._workQueue.put((self._W, self._K, doc_set, self._alpha, self._maxiters, self._thresh, self._useNewton))
        self._workQueue.join()
        return (self._alpha, self._posterior.value())


