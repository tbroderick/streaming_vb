# archived_dataset.py: Functions for loading documents from disk

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

import sys, re, string, time, threading, fileinput, mmap
import numpy as n

def loadDocs(path):
    #do this using memory-mapped io. faster? think so.
    print "Loading docs ..."
    #get number of lines 
    numLines = 0
    with open(path, "r+b") as f:
        m=mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        while(m.readline() != ''):
            numLines += 1
    print str(numLines) +" docs to load."
    docs = numLines *[None]
    #read the docs in
    with open(path, "r+b") as f:
        m=mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        i = 0;
        while(True):
            line = m.readline()
            if line == '':
                break
            #print line
            line = line.rstrip().lstrip()
            line = line[line.find(' ')+1:]
            split = line.split(" ")
            doc = (n.array([int(p.split(":")[0]) for p in split])
            ,n.array([int(p.split(":")[1]) for p in split]))
            #print doc
            #print 
            docs[i] = doc
            i += 1
    print "done."
    return docs



class Corpus:
    """
    Loads articles from a local corpus.
    """

    def __init__(self, path): # , min_words):
        #load the dataset from disk

        self._i = 0
        self._data = loadDocs(path)
        self._D = len(self._data)
        print "cache contains " + str(self._D) + " docs."

    def get_random_docs(self,n):
        """
        Loads n docs in parallel and returns lists
        of their contents.
        """
        docs = self._data[self._i:self._i+n]
        self._i += n
        return docs

if __name__ == '__main__':
        
        wr = Corpus("mult.dat",50)
        t0 = time.time()
        articles = wr.get_random_docs(10)

        t1 = time.time()
        print 'took %f' % (t1 - t0)
