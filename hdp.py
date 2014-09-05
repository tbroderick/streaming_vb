# the hdp class in python
# implements the truncated hdp model
# by chongw@cs.princeton.edu

import numpy as np
import scipy.special as sp
import os, sys, math, time
from itertools import izip
import random

meanchangethresh = 0.00001
random_seed = 999931111
np.random.seed(random_seed)
random.seed(random_seed)

def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])

def expect_log_sticks(sticks):
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks

class suff_stats:
    def __init__(self, T, size_vocab):
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_beta_ss = np.zeros((T, size_vocab))
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)

class hdp:
    ''' hdp model using john's new stick breaking'''
    def __init__(self, T, K,  size_vocab, eta, gamma, alpha, docs):
        ''' this follows the convention of the HDP paper'''
        ''' gamma, first level concentration ''' 
        ''' alpha, second level concentration '''
        ''' eta, the topic Dirichlet '''
        ''' T, top level truncation level '''
        ''' K, second level truncation level '''
        ''' size_vocab, size of vocab'''
        ''' gamma: prior on corpus level betas... (2,T-1) size array'''
        ''' alpha, topic level beta parameter '''
        ''' docs: a list of (wordids, wordcount) tuples'''
    
        self._docs = docs
        self._D = len(docs)
        

        self.m_T = T
        self.m_K = K # for now, we assume all the same for the second level truncation
        self.m_size_vocab = size_vocab

        self.m_beta = np.random.gamma(1.0, 1.0, (T, size_vocab)) * self._D*100/(T*size_vocab)
        self.m_eta = eta

        self.m_alpha = alpha
        #self.m_gamma = hdp_hyperparam.m_gamma_a/hdp_hyperparam.m_gamma_b
        self.m_gamma = gamma

        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] += gamma[0]
        self.m_var_sticks[1] += gamma[1]

    def save_topics(self, filename):
        f = file(filename, "w") 
        for beta in self.m_beta:
            line = ' '.join([str(x) for x in beta])  
            f.write(line + '\n')
        f.close()

    def doc_e_step(self, words, counts, ss, Elogbeta, Elogsticks_1st, var_converge, fresh=False):


        Elogbeta_doc = Elogbeta[:, words] 
        v = np.zeros((2, self.m_K-1))

        phi = np.ones((len(words), self.m_K)) * 1.0/self.m_K

        # the following line is of no use
        Elogsticks_2nd = expect_log_sticks(v)

        likelihood = 0.0
        old_likelihood = -1e1000
        converge = 1.0 
        eps = 1e-100
        
        iter = 0
        max_iter = 100
        #(TODO): support second level optimization in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi 
            if iter < 3 and fresh:
                var_phi = np.dot(phi.T, (Elogbeta_doc * counts).T)
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T, (Elogbeta_doc * counts).T) + Elogsticks_1st
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)

           # phi
            if iter < 3:
                phi = np.dot(var_phi, Elogbeta_doc).T
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)
            else: 
                phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            phi_all = phi * np.array(counts)[:,np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * counts))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < 0:
                print "warning, likelihood is decreasing!"
            
            iter += 1
            
        # update the suff_stat ss 
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, words] += np.dot(var_phi.T, phi.T * counts)

        return(likelihood)

    def optimal_ordering(self, ss):
        s = [(a, b) for (a,b) in izip(ss.m_var_sticks_ss, range(self.m_T))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        ss.m_var_sticks_ss[:] = ss.m_var_sticks_ss[idx]
        ss.m_var_beta_ss[:] = ss.m_var_beta_ss[idx,:]

    def do_m_step(self, ss):
        self.optimal_ordering(ss)
        ## update top level sticks 
        self.m_var_sticks[0] = ss.m_var_sticks_ss[:self.m_T-1] + self.m_gamma[0]#1.0
        var_phi_sum = np.flipud(ss.m_var_sticks_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma[1] 

        ## update topic parameters
        self.m_beta = self.m_eta + ss.m_var_beta_ss

    def seed_init(self):
        n = self._D
        ids = random.sample(range(n), self.m_T) 
        print "seeding with docs %s" % (' '.join([str(id) for id in ids]))
        for (id, t) in izip(ids, range(self.m_T)):
            doc = self._docs[id]
            self.m_beta[t] = np.random.gamma(1, 1, self.m_size_vocab) 
            self.m_beta[t,doc[0]] += doc[1]
    
    ## one iteration of the em
    def em(self, var_converge, fresh):
        ss = suff_stats(self.m_T, self.m_size_vocab)
        ss.set_zero()
        
        # prepare all needs for a single doc
        Elogbeta = dirichlet_expectation(self.m_beta) # the topics
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        likelihood = 0.0
        for doc in self._docs:
            likelihood += self.doc_e_step(doc[0],doc[1], ss, Elogbeta, Elogsticks_1st, var_converge, fresh=fresh)
        
        # collect the likelihood from other parts
        # the prior for gamma
        
        #log_gamma = np.log(self.m_gamma)

        #the W/sticks part 
        #likelihood += (self.m_T-1) * log_gamma

        #I am being lazy here - we need some terms to come from gamma, the parameter for the prior - but they don't factor into the convergence thr
        
        dig_sum = sp.psi(np.sum(self.m_var_sticks, 0))
        #likelihood += np.sum((np.array([1.0, self.m_gamma])[:,np.newaxis] - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
        likelihood += np.sum((self.m_gamma - self.m_var_sticks) * (sp.psi(self.m_var_sticks) - dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(self.m_var_sticks, 0))) - np.sum(sp.gammaln(self.m_var_sticks))
        
        # the beta part    
        likelihood += np.sum((self.m_eta - self.m_beta) * Elogbeta)
        likelihood += np.sum(sp.gammaln(self.m_beta) - sp.gammaln(self.m_eta))
        # original code has no sum here
        likelihood += np.sum(sp.gammaln(np.sum(self.m_eta,1)) - sp.gammaln(np.sum(self.m_beta, 1)))

        self.do_m_step(ss) # run m step
        return likelihood
               
    def train(self, iters, thr):
        oldB = self.em(1E-4,True)
        for i in range(iters-1):
            print i,oldB
            newB = self.em(1E-4,False)
            if (newB - oldB < thr):
                return
            oldB = newB
            

    def hdp_to_lda(self):
        # compute the lda almost equivalent hdp.
        # alpha
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        alpha = alpha * self.m_alpha

        #alpha = alpha * self.m_gamma
        
        # beta
        #beta_sum = np.sum(self.m_beta, axis=1)
        #beta = self.m_beta / beta_sum[:, np.newaxis]

        return (alpha, self.m_beta)

    

