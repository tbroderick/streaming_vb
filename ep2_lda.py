# Implementation of batch EP (expectation propagation) for LDA (Latent Dirichlet
# Allocation).
#
# This is the second version, following Minka's trick of only considering word tokens
# and performing a funky update.

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
import numpy as np
from scipy.special import gammaln, psi
from utils import *

np.random.seed(100000001)


class EP2_LDA:
    """
    Implements batch EP for LDA, second version (fake EP, following Minka's trick).
    """

    def __init__(self, W, K, docs, alpha, eta, useNewton):
        """
        Arguments:
        K: Number of topics
        W: Number of words in the vocabulary
        docs: a list of {tuple of (wordsids, wordcts)}
        alpha: Hyperparameter for prior on weight vectors theta, a scalar (the vector alpha has all identical entries)
        eta: Hyperparameter for prior on topics beta, either a scalar or a K-by-W array
        useNewton: Boolean, if true we use Newton's method for solving Dirichlet moment-matching, else we use approximate MLE
        """
     
        self._K = K
        self._docs = docs
        self._W = W
        self._D = len(docs)
        self._alpha = alpha
        if np.isscalar(eta):
           self._eta = eta * np.ones((self._K, self._W))
        else:
            self._eta = eta
        self._useNewton = useNewton

        # For each document, get list of words appearing in the documents and their counts
        (self._wordids, self._wordcts) = unzipDocs(docs)

        # Get number of distinct words in each document
        self._Vd = [len(ids) for ids in self._wordids]

        # Initialize the variational distribution q(theta | gamma)
        # with gamma_d = alpha + \sum_{v \in V_d} n_dv * zeta_dv
        # by setting all zeta_dv = 0.
        # Here V_d is the set of unique words appearing in document d, and n_dv is
        # the number of times word v appears in document d.
        # zeta is a list of D matrices, with zeta[d] being a |V_d|-by-K matrix.
        self._zeta = list()
        for d in range(0, self._D):
            self._zeta.append(np.zeros([self._Vd[d], self._K]))

        # Initialize the variational distribution q(beta | lambda)
        # with lambda_k = eta_k + \sum_d \sum_{v \in V_d} n_dv * omega_kdv
        # by setting all omega_kdv = 0.
        # omega is a list of K entries, with omega[k] being a list of D matrices,
        # with omega[k][d] being a |V_d|-by-W matrix.
        self._omega = list()
        for k in range(0, self._K):
            omega_k = list()
            for d in range(0, self._D):
                omega_k.append(np.zeros([self._Vd[d], self._W]))
            self._omega.append(omega_k)

        # Initialize lambda and gamma to be equal to the prior eta and alpha
        # (Note: This assumes omega and zeta to be initialized to 0)
        self._lambda = np.empty_like(self._eta)  # K-by-W array
        self._lambda[:] = self._eta  # copy eta
        self._gamma = self._alpha * np.ones([self._D, self._K])



    def train(self, maxiters, thr, silent = False):
        # Prepare placeholder for old values of lambda and gamma
        old_lambda = np.empty_like(self._lambda)
        old_gamma = np.empty_like(self._gamma)

        # One iteration is going over all words in all documents
        for iteration in range(0,maxiters):
            # First make a copy of the old values to check for convergence later
            old_lambda[:] = self._lambda
            old_gamma[:] = self._gamma

            # Update omega and zeta. This method keeps lambda and gamma up to date with omega and zeta
            self.update_omega_zeta()

            # Compute new gamma, and check for convergence
            delta = self.compute_diff(old_lambda, old_gamma)
            # if not silent:
            #     print "\tIter " + str(iteration+1) +"/" + str(maxiters) + ": delta = " + str(delta) + " (thr = " + str(thr) + ")" #not pythonic
            if delta < thr:
                # print "Batch EP2 converged in " + str(iteration+1) + " iterations, delta = " + str(delta) + " (thr = " + str(thr) + ", avg_lambda = " + str(np.mean(self._lambda)) + ", avg_gamma = " + str(np.mean(self._gamma)) + ")" 
                # print ""
                break
            if iteration >= maxiters - 1:
                print "Batch EP2 did not converge after " + str(iteration+1) + " iterations, delta = " + str(delta) + " (thr = " + str(thr) + ", avg_lambda = " + str(np.mean(self._lambda)) + ", avg_gamma = " + str(np.mean(self._gamma)) + ")"
                print ""
        return self._lambda


    def update_omega_zeta(self):
        """
        Iterates over all words in all documents to iteratively update omega and zeta, and keep lambda and gamma current.
        """
        for d in range(0, self._D):
            # print "\tAt doc " + str(d+1) + "/" + str(self._D) + ": N_d =  " + str(self._N[d])
            for v in range(0, self._Vd[d]):  # v indexes the set V_d
                # print "\t\td = " + str(d+1) + "/" + str(self._D) + ", n = " + str(n+1) + "/" + str(self._N[d])

                # Compute rho_d = gamma_d - zeta_dv
                rho_d = self._gamma[d,:] - self._zeta[d][v,:]  # array of length K

                # For 1 <= k <= K, compute tau_k = lambda_k - omega_kdv
                tau = np.empty_like(self._lambda)
                for k in range(0, self._K):
                    tau[k,:] = self._lambda[k,:] - self._omega[k][d][v,:]

                # If some components of rho_d or tau_k are non-positive, skip this step
                if (rho_d.min() <= 0 or tau.min() <= 0):
                    print "\t\tSkipped (d,v) = (" + str(d) + "," + str(v) + "): min(rho_d) = " + str(rho_d.min()) + ", min(tau) = " + str(tau.min())
                    continue

                # Compute \sum_k \rho_dk and \sum_w \tau_kw, for 1 <= k <= K
                sum_rho_d = np.sum(rho_d)  # scalar
                sum_tau = np.sum(tau, 1)  # array of length K

                # The current word w_dn
                wdn = self._wordids[d][v]  # this is an integer between 0 and W-1 (size of the vocabulary)

                # Compute C_k, the (normalized) mixing proportions
                C = (rho_d * tau[:,wdn]) / sum_tau  # elementwise operation
                C = C / np.sum(C)  # nonnegative array of length K, summing to 1

                # Compute E[theta_dk] and E[theta_dk^2] with respect to the
                # approximate posterior distribution (which is a mixture of Dirichlet with
                # mixing proportion specified by C)
                E_theta_d = (rho_d + C) / (sum_rho_d + 1)
                E_theta_d2 = (rho_d * (rho_d + 1) + 2 * C * (rho_d + 1)) / ((sum_rho_d + 1) * (sum_rho_d + 2))

                # If we use Newton's method, we need to compute E[log theta_dk] as well
                if (self._useNewton):
                    E_log_theta_d = (1 - C) * psi(rho_d) + C * psi(rho_d + 1) - psi(sum_rho_d + 1)
                    new_gamma_d = dirichlet_mle_newton(E_theta_d, E_theta_d2, E_log_theta_d)
                else:
                    new_gamma_d = (sum(E_theta_d - E_theta_d2) / sum(E_theta_d2 - E_theta_d ** 2)) * E_theta_d

                # Then update zeta and gamma
                self._zeta[d][v,:] += (new_gamma_d - self._gamma[d,:]) / self._wordcts[d][v]
                self._gamma[d,:] = new_gamma_d

                # For each k = 1, ..., K, compute E[beta_kv] and E[beta_kv^2] with respect to the
                # approximate posterior distribution, do moment-matching for beta_k, and update lambda_k and omega_k
                for k in range(0, self._K):
                    C_k = C[k]
                    tau_k = tau[k,:]
                    sum_tau_k = sum_tau[k]
                    tau_k_wdn = tau_k[wdn]

                    E_beta_k = tau_k / (sum_tau_k + 1) + (1 - C_k) * tau_k / (sum_tau_k * (sum_tau_k + 1))
                    E_beta_k[wdn] += C_k / (sum_tau_k + 1)

                    E_beta2_k = tau_k * (tau_k + 1) / ((sum_tau_k + 1) * (sum_tau_k + 2)) \
                                  + 2 * (1 - C_k) * tau_k * (tau_k + 1) / (sum_tau_k * (sum_tau_k + 1) * (sum_tau_k + 2))
                    E_beta2_k[wdn] += 2 * C_k * (tau_k_wdn + 1) / ((sum_tau_k + 1) * (sum_tau_k + 2))

                    # If we use Newton's method, we need to compute E[log beta_kv] as well
                    if (self._useNewton):
                        E_log_beta_k = psi(tau_k) - (1 - C_k) * psi(sum_tau_k) - C_k * psi(sum_tau_k + 1)
                        E_log_beta_k[wdn] += C_k * (psi(tau_k_wdn + 1) - psi(tau_k_wdn))
                        new_lambda_k = dirichlet_mle_newton(E_beta_k, E_beta2_k, E_log_beta_k)
                    else:
                        new_lambda_k = (sum(E_beta_k - E_beta2_k) / sum(E_beta2_k - E_beta_k ** 2)) * E_beta_k

                    # Then update omega
                    self._omega[k][d][v,:] += (new_lambda_k - self._lambda[k,:]) / self._wordcts[d][v]
                    self._lambda[k,:] = new_lambda_k


    def  compute_diff(self, old_lambda, old_gamma):
        """
        Compute the average component-wise change
        in the values of current and old lambda and gamma.
        This method assumes self._lambda and self._gamma are up-to-date with
        the current values of self._omega and self._zeta.
        """
        gamma_diff = np.mean(abs(self._gamma - old_gamma))
        lambda_diff = np.mean(abs(self._lambda - old_lambda))
        return (0.5 * (gamma_diff + lambda_diff))
