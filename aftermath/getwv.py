import os

def token2dir(token, dirdepth, noletter='none'):
	assert isinstance(token, basestring), 'token is not a string'

	dirs = [noletter]*dirdepth
	for i in range(min(len(token),dirdepth)):
		dirs[i] = token[i]
	return('/'.join(dirs))

def tokenvec(tokens, topdir, dirdepth, d=None, missing0=False, fname='tokens.txt', noletter='none', nowarnings=False, lower=True):
	if(not hasattr(tokens, '__iter__')): tokens = [tokens]
	if missing0:
		assert isinstance( d, ( int, long ) ), 'must provide an integer for d if missing0 is True'
	retlist = []
	missing = []
	for token in tokens:
		# if empty list is provided, replace empty with ''
		if token is None: token = ''
		if lower: token = token.lower()
		worddir = token2dir(token, dirdepth, noletter)
		found = False
		try:
			with open('/'.join([topdir, worddir, fname])) as f:
				for line in f:
					if token+' ' in line:
						retlist += [[float(x) for x in line[:-1].split(' ')[1:]]]
						found = True
						break
				
		except IOError, e:
			if e.errno==2:
				# do nothing
				found = False
			else:
				print "Unexpected IOerror"
				raise
		# didn't find anything
		if not found:
			missing += [token]
			if missing0:
				retlist += [[0]*d]
			else:
				retlist += [[]]
	if (not nowarnings) and len(missing) > 0:
		print "Warning - could not find tokens: '%s'" % "','".join(missing)
	return retlist

def tokenvectest():
	tokens = ['understand','!!','!','abdfcd', 'UNDERstand']
	topdir = '../data/twitter/50d'
	fname = 'tokens.txt'
	dirdepth = 3

	print tokenvec(tokens[0], topdir, dirdepth, fname=fname)
	print tokenvec(tokens[1], topdir, dirdepth, fname=fname)
	print tokenvec(tokens, topdir, dirdepth, fname=fname)

#tokenvectest()