import os
import pdb

# This file is used to hash
# the tokens from glove.twitter.27B.100d.txt to
# folders based on the initial letters of the
# token. This makes it much faster to search
# for the vectors associated with the tokens later.
# The file fname and directory topdir must exist in
# the directory fpath.
# It takes a token from the beginning of each line
# of fpath and uses the first ndir characters
# of the token to create ndir nested directories
# in fpath/topdir.
# It takes a while to run.

def write2file(x, fname, mode='wb'):
	if not os.path.exists(os.path.dirname(fname)):
		try:
			os.makedirs(os.path.dirname(fname))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(fname, mode) as f:
		f.write(x+'\n')

fpath = '../data/twitter'
fname = 'glove.twitter.27B.200d.txt'
topdir = '200d'
fout = 'tokens.txt'
ndir = 3
ctr = 0
verbose = False

with open('/'.join([fpath,fname]),'r') as f:
    for x in f:
        x = x.rstrip()
        ascii_check = all(ord(c) < 128 for c in x)
        if (not x) or (not ascii_check): continue
        dtup = []
        for i in range(ndir):
        	if x[i] == ' ':
        		for j in range(i,ndir): dtup += ['none']
        		break
        	dtup += [x[i]]

        writef = '/'.join([fpath,topdir]+dtup+[fout])

        write2file(x, writef, 'a+')
        if verbose:
        	print writef
        	print x
        ctr+=1
        #if ctr > 10: break
