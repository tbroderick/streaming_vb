======================================
= README
======================================

Contents
1. History and licensing information
2. Data format
3. How to run

======================================
1. History and licensing information
======================================

This code is largely the same as, and adapted from, the online VB (aka stochastic variational Bayes) code of
Matthew D. Hoffman, Copyright (C) 2010
found here: http://www.cs.princeton.edu/~blei/downloads/onlineldavb.tar
and also of 
Chong Wang, Copyright (C) 2011
found here: http://www.cs.cmu.edu/~chongw/software/onlinehdp.tar.gz
The GPL license is inherited from that code.

Adapted by: Nick Boyd, Tamara Broderick, Andre Wibisono, Ashia C. Wilson

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

======================================
2. Data format
======================================

Each data set should consist of three files. In what follows, NAME represents the name of the corpus used as a prefix in filenames. In our experiments we used NAME equal to "wiki" or "nature". The three files should be:
1. NAME_vocab.txt: A file with one vocabulary word from the corpus per line.
2. NAME_train.txt: The training data for the corpus. Each line represents a document. Each line should be in the format:
U_D I_1:N_1 I_2:N_2 ... I_M:N_M
where U_D is the number of unique vocabulary words in this document, I_m is the index of the mth unique vocabulary word in the NAME_vocab.txt file, and N_m is the number of times this word occurs in this document. There is a space in between each index-count pair and a space after the count of unique vocabulary words.
3. NAME_test.txt: The test data for the corpus. This file is in the same format as the training data.

======================================
3. How to run
======================================

Below are some example use cases.

To run single-thread streaming variational Bayes on a data set with name NAME:
$ python onlinewikipedia.py --algorithmname=filtering --corpus=NAME --batchsize=32768 --eta=0.01 --max_iters=100 --threshold=1

To run synchronous, distributed, streaming variational Bayes on a data set with name NAME with 16 processors:
$ python onlinewikipedia.py --algorithmname=filtering --corpus=NAME
--batchsize=32768 --eta=0.01 --max_iters=100 --threshold=1 --numthreads=16

To run asynchronous, distributed, streaming variational Bayes on a data set with name NAME with 16 processors:
$ python onlinewikipedia.py --algorithmname=filtering --corpus=NAME
--batchsize=32768 --async_batches_per_eval=4 --eta=0.01 --max_iters=100 --threshold=1 --numthreads=16

To run the sufficient statistics algorithm on a data set with name NAME:
$ python onlinewikipedia.py --algorithmname=ss --corpus=NAME --batchsize=32768 --eta=0.01
