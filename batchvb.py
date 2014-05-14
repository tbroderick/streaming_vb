# batchvb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with variational Bayes (VB).

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
from scipy.special import gammaln, psi
from utils import *

n.random.seed(100000001)


class BatchLDA:
    """
    Implements batch VB for LDA.
    """

    def __init__(self, W, K, docs, alpha, eta, useHBBBound):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        doc: a tuple of (wordsids, wordcts)
        alpha: Hyperparameter for prior on weight vectors theta, a scalar (the vector alpha has all identical entries)
        eta: Hyperparameter for prior on topics beta, either a scalar or a K-by-W array
        useHBBBound: Boolean, if false, use change in parameters as stopping criterion
        """
        self._useHBBBound = useHBBBound
        self._K = K
        self._docs = docs
        self._W = W
        self._D = len(docs)
        self._alpha = alpha
        if n.isscalar(eta):
           self._eta = eta * n.ones((self._K, self._W))
        else:
            self._eta = eta
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self.set_lambda(1*n.random.gamma(100., 1./100., (self._K, self._W)))
        #self._gamma = 1*n.random.gamma(100., 1./100., (self._D, self._K)) #now we can do warmstart on gamma... should be MUCH faster?

        #do this once... doesn't really save much time...
        (self._wordids, self._wordcts) = unzipDocs(docs)

    def train(self, maxiters, thr):
        
        if (not self._useHBBBound):
            oldB = n.empty_like(self._lambda)
            oldB[:] = self._lambda
        else:
            oldB = float("inf") 

        for iteration in range(0,maxiters):
            newB = self.update_lambda()  # equal to lambda matrix if not useHBBBound, else ELBO-like quantity
            change = abs(oldB - newB).max()  # change in stuff

            # print str(iteration) + "/" + str(maxiters) + "; oldB: " + str(n.mean(oldB)) + ", newB: " + str(n.mean(newB)) + ", change: " + str(change) + " (thr: " + str(thr) + ")"

            # Check for convergence
            if (change < thr):
                # print "\tBatchVB converged in " + str(iteration+1) + "/" + str(maxiters) + " iterations, change: " + str(change) + ", thr: " + str(thr)
                break
            if iteration >= maxiters - 1:
                print "\tBatchVB did not converge after " + str(maxiters) + " iterations, change: " + str(change) + ", thr: " + str(thr)
            oldB = newB
        return self._lambda


    def set_lambda(self, lam):
        self._lambda = lam
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self):
        """
        Returns the sufficient statistics needed to update lambda.
        """
        self._gamma = 1*n.random.gamma(100., 1./100., (self._D, self._K)) 
        Elogtheta = dirichlet_expectation(self._gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, self._D):
            # These are mostly just shorthand (but might help cache locality)
            ids = self._wordids[d]
            cts = self._wordcts[d]
            gammad = self._gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            self._gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return sstats 

    def update_lambda(self):
        """
        First does an E step, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # Do an E step to update gamma, phi | lambda. This also returns the information about phi that
        # we need to update lambda.
        sstats = self.do_e_step()
        # Estimate held-out likelihood for current values of lambda.
        if (not self._useHBBBound):
            bound = self._eta + sstats  # the new lambda, set below
        else:
            bound = self.approx_bound()  # the ELBO-like bound
        # Update lambda based on documents.
        self.set_lambda(self._eta + sstats)
        self._updatect += 1

        return bound  # lambda matrix if not useHBBBound, else ELBO-like bound




    def approx_bound(self):
        """
        MUST FIRST UPDATE GAMMA (by calling do_e_step)
        The variational bound over all documents. gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents.
        """
        
        score = 0
        Elogtheta = dirichlet_expectation(self._gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, self._D):
            gammad = self._gamma[d, :]
            ids = self._wordids[d]
            cts = n.array(self._wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - self._gamma)*Elogtheta)
        score += n.sum(gammaln(self._gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(self._gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        #original code has no sum here! ah. 
        score = score + n.sum(gammaln(n.sum(self._eta,1)) - 
                              gammaln(n.sum(self._lambda, 1)))
        perwordbound = score / sum(map(sum, self._wordcts))
        return n.exp(-perwordbound)
