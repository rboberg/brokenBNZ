import os
import pdb

def write2file(x, fname, mode='wb'):
	if not os.path.exists(os.path.dirname(fname)):
		try:
			os.makedirs(os.path.dirname(fname))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	with open(fname, mode) as f:
		f.write(x+'\n')

fname = 'glove.twitter.27B.50d.txt'
topdir = '50d'
fout = 'tokens.txt'
ndir = 3
ctr = 0
verbose = False

with open(fname,'r') as f:
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
		
        writef = topdir + '/' +'/'.join(dtup) + '/' + fout
        
        write2file(x, writef, 'a+')
        if verbose:
        	print writef
        	print x
        ctr+=1
        #if ctr > 10: break
