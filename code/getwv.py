import os

def token2dir(token, dirdepth, noletter='none'):
	assert isinstance(token, basestring), 'token is not a string'

	dirs = [noletter]*dirdepth
	for i in range(min(len(token),dirdepth)):
		dirs[i] = token[i]
	return('/'.join(dirs))

def tokenvec(tokens, topdir, dirdepth, fname='tokens.txt', noletter='none', nowarnings=False):
	if(not hasattr(tokens, '__iter__')): tokens = [tokens]
	retlist = []
	missing = []
	for token in tokens:
		worddir = token2dir(token, dirdepth, noletter)
		try:
			with open('/'.join([topdir, worddir, fname])) as f:
				found = False
				for line in f:
					if token+' ' in line:
						retlist += [[float(x) for x in line[:-1].split(' ')[1:]]]
						found = True
						break
				# didn't find anything
				if not found:
					missing += [token]
					retlist += [[]]
		except IOError, e:
			if e.errno==2:
				missing += [token]
				retlist += [[]]
			else:
				print "Unexpected IOerror"
				raise
	if (not nowarnings) and len(missing) > 0:
		print "Warning - could not find tokens: '%s'" % "','".join(missing)
	return retlist

def tokenvectest():
	tokens = ['understand','!!','!','abdfcd']
	topdir = '50d'
	fname = 'tokens.txt'
	dirdepth = 3

	print tokenvec(tokens[0], topdir, dirdepth, fname)
	print tokenvec(tokens[1], topdir, dirdepth, fname)
	print tokenvec(tokens, topdir, dirdepth, fname)