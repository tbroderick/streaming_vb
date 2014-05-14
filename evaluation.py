# evaluation.py: Code for checking performance on the test data

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

import cPickle, string, getopt, sys, random, time, re, pprint
import numpy as np
import onlineldavb, batchvb
#import wikirandom
import copy,math
from scipy.special import gammaln, psi

from utils import *

#adapted mostly from chong's excellent HDP code...

def evaluate(docs, lda_alpha, lam, usePtEst): 
  if (usePtEst):
    lam_sum = np.sum(lam, axis=1)
    lda_beta = lam / lam_sum[:, np.newaxis]
    return evaluateBeta(docs,lda_alpha, lda_beta)
  else:
    return evaluateLambda(docs, lda_alpha, lam)

def evaluateBeta(docs, lda_alpha, lda_beta): 
  k = np.shape(lda_beta)[0]
  numwords = 0.0
  test_score = 0.0
  test_score_split = 0.0
  c_test_word_count_split = 0
  for docI in range(0,len(docs)):
    (wordids, wordcts) = (docs[docI][0], docs[docI][1])
    # (likelihood, gamma) = lda_e_step((wordids, wordcts), lda_alpha, k, lda_beta)
    # test_score += likelihood
    (likelihood, count, gamma) = lda_e_step_split((wordids, wordcts), lda_alpha, k, lda_beta)
    test_score_split += likelihood
    c_test_word_count_split += count
    numwords += sum(wordcts)
  return (test_score / numwords, test_score_split / c_test_word_count_split)


def evaluateLambda(docs, lda_alpha, lam): 
  k = np.shape(lam)[0]
  numwords = 0.0
  test_score = 0.0
  test_score_split = 0.0
  c_test_word_count_split = 0
  for docI in range(0,len(docs)):
    (wordids, wordcts) = (docs[docI][0], docs[docI][1])
    # (likelihood, gamma) = lda_e_step_full((wordids, wordcts), lda_alpha, k, lam)
    # test_score += likelihood
    (likelihood, count, gamma) = lda_e_step_split_full((wordids, wordcts), lda_alpha, k, lam)
    test_score_split += likelihood
    c_test_word_count_split += count
    numwords += sum(wordcts)
  return (test_score / numwords, test_score_split / c_test_word_count_split)


def lda_e_step((words, counts), alpha, k, Elogbeta, max_iter=100):
    gamma = np.ones(k)  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    expElogbeta = Elogbeta[:, words]
    phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
    counts = np.array(counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm,  expElogbeta.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))
    return (likelihood, gamma)


def lda_e_step_full((words, counts), alpha, k, lam, max_iter=100):
    gamma = np.ones(k)  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    expElogbeta = np.exp(dirichlet_expectation(lam))[:, words]
    phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
    counts = np.array(counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm,  expElogbeta.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))
    return (likelihood, gamma)


def lda_e_step_split((words, counts), alpha, k, beta, max_iter=100):
    length = len(words)
    half_len = int(len(words) / 2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < length]
   
    # split the document
    words_train = [words[i] for i in idx_train]
    counts_train = [counts[i] for i in idx_train]
    words_test = [words[i] for i in idx_test]
    counts_test = [counts[i] for i in idx_test]

    gamma = np.ones(k)  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)



def lda_e_step_split_full((words, counts), alpha, k, lam, max_iter=100):
    length = len(words)
    half_len = int(len(words) / 2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < length]
   
    # split the document
    words_train = [words[i] for i in idx_train]
    counts_train = [counts[i] for i in idx_train]
    words_test = [words[i] for i in idx_test]
    counts_test = [counts[i] for i in idx_test]

    gamma = np.ones(k)  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    expElogbeta = np.exp(dirichlet_expectation(lam))[:, words_train]
    phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  expElogbeta.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    lam_sum = np.sum(lam, axis=1)
    Ebeta = lam / lam_sum[:, np.newaxis]
    Ebeta_test = Ebeta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, Ebeta_test) + 1e-100))

    return (score, np.sum(counts), gamma)


