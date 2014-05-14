#!/usr/bin/python

# Main function

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
import onlineldavb, os, utils, parallelfiltering, onlinevb, filtering, asynchronous
import evaluation
from optparse import OptionParser
import archived_dataset

def parse_args():
    parser = OptionParser()
    parser.set_defaults(K=100, D=-1, corpus = "nature", alpha = 1.0, eta = 1.0, batchsize=1024,
        async_batches_per_eval=5, 
        max_iters=100, threshold=1.0, numthreads=1, tau0=1.0, kappa=0.9, useHBBBound = True,
        minNumPtsPerEval = -1, expGrowthEval=1)

    parser.add_option("--algorithmname", type="string", dest="algorithmname",
                    help="which algorithm to use {filtering, filtering_ep, hbb, ss}")
    parser.add_option("--K", type="int", dest="K",
                    help="number of topics")
    parser.add_option("--D", type="int", dest="D",
                    help="number of documents for HBB")
    parser.add_option("--corpus", type="string", dest="corpus",
                    help="name of corpus {nature, wiki}")
    parser.add_option("--batchsize", type="int", dest="batchsize",
                    help="batchsize")
    parser.add_option("--async_batches_per_eval", type="int", dest="async_batches_per_eval",
                    help="number of batches between evaluation for async filtering (NOTE THIS AFFECTS PERFORMANCE IF SMALL!)")
    parser.add_option("--max_iters", type="int", dest="max_iters",
                    help="max_iters for batchvb subroutine for filtering")
    parser.add_option("--threshold", type="float", dest="threshold",
                    help="threshold for batchvb subroutine")
    parser.add_option("--numthreads", type="int", dest="numthreads",
                    help="number of threads to use for filtering algorithms")
    parser.add_option("--eta", type="float", dest="eta",
                    help="eta parameter for beta (all the same)")
    parser.add_option("--tau0", type="float", dest="tau0",
                    help="tau0 for hbb")
    parser.add_option("--kappa", type="float", dest="kappa",
                    help="kappa for hbb")
    parser.add_option("--async", dest="async", action='store_true',
                    help="Use asynchronous parallel filtering ", default=False)
    parser.add_option("--useNewton", dest="useNewton", default=False, action='store_true',
                    help="Use newtons method for moment-matching in EP")
    parser.add_option("--usePtEst", dest="usePtEst", default=False, action='store_true',
                    help="Use point estimate of beta for computing log predictive probability")
    parser.add_option("--minNumPtsPerEval", type="int", dest="minNumPtsPerEval",
                    help="minimum number of points to be processed between evaluation")
    parser.add_option("--expGrowthEval", type="int", dest="expGrowthEval",
                    help="double the number of points processed between evaluations at each evaluation? (0 for no; 1 for yes)")

    (options, args) = parser.parse_args()
    if options.algorithmname is None:
        parser.print_help()
        exit(-1)
    return options 

def main():
    """
    Analyzes specified documents.
    """
    options = parse_args()
    print options

    # we assume there exist three files: 
    # a vocab file (corpus_vocab.dat)
    # a training file (corpus_train.dat)
    # a validation file (corpus_test.dat)

    corpus = options.corpus

    # vocab file
    W = len(open(corpus + "_vocab.dat", 'r').readlines())

    # validation file
    validation_filename = corpus + "_test.dat"

    wikirandom = archived_dataset.Corpus(corpus + "_train.dat") # should be _train.dat
    # else:
    #     import wikirandom

    #load a held-out set
    validation_docs = archived_dataset.loadDocs(validation_filename)
    algorithmname = options.algorithmname

    # the second tells us the batch size
    batchsize = options.batchsize

    # the third tells us a list of number of threads to run. (each will be run sequentially)
    numthreads = options.numthreads

    # number of documents
    trueD = wikirandom._D

  
    if(algorithmname == "hbb"):
        if options.D == -1:
            D = trueD # number of documents to know in advance
        else:
            D = options.D


    # #prior for topics (ANDRE: this is now a parameter)
    # eta = 1.
    eta = options.eta
    
    # The total number of documents
    #D = 3.3e6 (used to be number in Wikipedia; now an argument)

    # The number of topics
    K = options.K
    alpha = 1./K #* numpy.ones(K)
    batchsize = options.batchsize
    
    if (algorithmname == "hdp_filtering"):
        alg = filtering.HDPFiltering(W,eta, options.max_iters,options.threshold*1E-6, T = 300, K = 30)

    if (algorithmname == "ss"):
        if (numthreads == 1):
            alg = filtering.Filtering(W, K, alpha, eta, 1, True, 0.1) # note: last two args shouldn't matter
        else:
			# NOT REALLY SUPPORTED!
            alg =  parallelfiltering.ParallelFiltering(W, K, alpha, eta, 1, 0.1,True,options.numthreads)
			
    if (algorithmname == "filtering"):
        #maxiters = 15
        if (numthreads == 1):
            alg = filtering.Filtering(W, K, alpha, eta, options.max_iters, options.useHBBBound, options.threshold)
        else:
            if (options.async):
                alg = asynchronous.ParallelFiltering(W, K, alpha, eta, options.max_iters, options.threshold, options.useHBBBound, options.batchsize, options.numthreads)
 
                batchsize = batchsize * options.async_batches_per_eval * options.numthreads
            else:
                alg =  parallelfiltering.ParallelFiltering(W, K, alpha, eta, options.max_iters, options.threshold, options.useHBBBound, options.numthreads, options.batchsize)
                batchsize = batchsize * options.numthreads

    if (algorithmname == "hbb"):
        #default: tau0 = 1024; kappa = 0.7
        # paper says: kappa = 0.5; tau0 = 64; S (minibatch size) = 4096
        # alg = onlineldavb.OnlineLDA(W, K, D, alpha, 1./K, options.tau0, options.kappa)  # the original code for NIPS submission, eta = 1/K
        alg = onlineldavb.OnlineLDA(W, K, D, alpha, eta, options.tau0, options.kappa)

    # EP for LDA
    if (algorithmname == "filtering_ep"):
        if (numthreads == 1):
            alg = filtering.FilteringEP(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton)
        else:
            if (options.async):
                alg = asynchronous.ParallelFilteringEP(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton, options.batchsize, options.numthreads)
                batchsize = batchsize * options.async_batches_per_eval * options.numthreads
            else:
                alg = parallelfiltering.ParallelFilteringEP(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton, options.numthreads, options.batchsize)
                batchsize = batchsize * options.numthreads

    # Fake EP for LDA (?) -- to be removed eventually since it's worse than true EP
    if (algorithmname == "filtering_ep2"):
        if (numthreads == 1):
            alg = filtering.FilteringEP2(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton)
        else:
            if (options.async):
                alg = asynchronous.ParallelFilteringEP2(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton, options.batchsize, options.numthreads)
                batchsize = batchsize * options.async_batches_per_eval * options.numthreads
            else:
                alg = parallelfiltering.ParallelFilteringEP2(W, K, alpha, eta, options.max_iters, options.threshold, options.useNewton, options.numthreads, options.batchsize)
                batchsize = batchsize * options.numthreads
    

    # Specify the minimum number of points to be processed before we run the evaluation code, since evaluation is expensive
    minNumPtsPerEval = options.minNumPtsPerEval
    expGrowthEval = options.expGrowthEval
    if (minNumPtsPerEval <= 0):
        if (corpus == "nature"):  # 351K docs
            minNumPtsPerEval = 512 #1e3
        elif (corpus == "wiki"):  # 3.6M docs
            minNumPtsPerEval = 512 #1e3 #2e4
        else:
            minNumPtsPerEval = int(trueD / 1000)

    print "Using algorithm: " + str(alg)
    recordedData = []
    totalTime = 0.0
    totalDownloadingTime = 0.0
    iters = int(trueD / batchsize) + 1
    numPtsProc = 0  # number of points processed since last evaluation
    for iteration in range(iters):
        # Get some articles
        start = time.time()
        docset = wikirandom.get_random_docs(batchsize)
        totalDownloadingTime += time.time() - start
        start = time.time()
        (alg_alpha, alg_lam) = alg.update_lambda(docset)
        iter_time = time.time() - start
        totalTime += iter_time
        numPtsProc += batchsize  # we have processed this many more points
        if (numPtsProc >= minNumPtsPerEval or iteration == iters-1):  # evaluate if we have processed enough points, or this is the last iteration
            numPtsProc = 0  # reset the counter
            # The following is just the usual evaluation code from before
            start = time.time()
            (perplex, split) = evaluation.evaluate(validation_docs, alg_alpha, alg_lam, options.usePtEst)
            testTime = time.time() - start
            print str(iteration+1) + "/" + str(iters) + " " + str(alg) + " (%g, %g): held-out perplexity estimate = %f, %f" % (iter_time, testTime, perplex, split)
            recordedData += [((iteration+1)*batchsize, totalTime, totalDownloadingTime, perplex, split)]  # also save perplexity now!
            if (algorithmname in ["hbb", "filtering", "filtering_ep", "filtering_ep2"]):
    	        outfile = corpus + "_" + str(alg) + "_" + str(batchsize) + "_eta" + str(eta)  # need to distinguish eta now
            else:
    	        outfile = corpus + "_" + algorithmname + "_" + str(options.batchsize) + "_" + str(options.numthreads) + "_eta" + str(eta)
            numpy.save(outfile, recordedData)

            if (expGrowthEval):
				# double the number of points to the next evaluation
    	        minNumPtsPerEval = minNumPtsPerEval * 2
        else:
            print str(iteration+1) + "/" + str(iters) + " " + str(alg) + " (%g)" % (iter_time)

        if (iteration == iters-1):
            # save final lambda matrix
            if (algorithmname in ["hbb", "filtering", "filtering_ep", "filtering_ep2"]):
                topics_outfile = "topics_" + corpus + "_" + str(alg) + "_" + str(batchsize) + "_eta" + str(eta)  # need to distinguish eta now
            else:
                topics_outfile = "topics_" + corpus + "_" + algorithmname + "_" + str(options.batchsize) + "_" + str(options.numthreads)
            numpy.save(topics_outfile, alg_lam)

	# asynchronous filtering needs to terminate its workers
    if (algorithmname == "filtering"):
        if (numthreads > 1):
            if (options.async):
                alg.shutdown()

    print "DONE!"

if __name__ == '__main__':
    main()
