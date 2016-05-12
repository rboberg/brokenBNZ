# attempt at python version of this ruby code:
# http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
# to preprocess reddit pizza request data for word vectors

import re
import pdb

#text = "http://www.tutorialspoint.com/ruby/ruby_regular_expressions.htm www.website.com :) :| :( in/out :p #awesome oh MY godddd!!! i <3 it 1052.2 times"

def glove_prep(text):

	eyes = "[8:=;]"
	nose = "['`\-]?"

	url = "https?:\/\/\S+|www\.(\w+\.)+\S*"
	assert re.sub(url, '<URL>', "https://www.tutorialspoint.com/ruby/ruby_regular_expressions.htm") == '<URL>'
	assert re.sub(url, '<URL>', "www.website.com/page") == '<URL>'

	text = re.sub(url, '<URL>', text)

	assert re.sub('/', ' / ' , "in/out") == 'in / out'
	text = re.sub("/", ' / ', text)# Force splitting words appended with slashes (once we tokenized the URLs, of course)

	assert re.sub('@\w+', '<USER>', '@me not @you') == '<USER> not <USER>'

	smiles = "{eyes!s}{nose!s}[)d]+|[(d]+{nose!s}{eyes!s}".format(eyes=eyes, nose=nose)
	assert re.sub(smiles, "<SMILE>", ":) 8-) (';") == '<SMILE> <SMILE> <SMILE>'
	text = re.sub(smiles, '<SMILE>', text)

	lols = "{eyes!s}{nose!s}p+".format(eyes=eyes, nose=nose)
	assert re.sub(lols, "<LOLFACE>", ":p :-ppp") == '<LOLFACE> <LOLFACE>'
	text = re.sub(lols, '<LOLFACE>', text)

	sads = "{eyes!s}{nose!s}\(+|\)+{nose!s}{eyes!s}".format(eyes=eyes, nose=nose)
	assert re.sub(sads, "<SADFACE>", "8-( ):") == '<SADFACE> <SADFACE>'
	text = re.sub(sads, '<SADFACE>', text)

	neuts = "{eyes!s}{nose!s}[\/|l*]".format(eyes=eyes, nose=nose)
	assert re.sub(neuts, "<NEUTRALFACE>", ":| 8-/ :l") == '<NEUTRALFACE> <NEUTRALFACE> <NEUTRALFACE>'
	text = re.sub(neuts, '<NEUTRALFACE>', text)

	heart = '<3'
	assert re.sub(heart, "<HEART>", "I <3 you") == 'I <HEART> you'
	text = re.sub(heart, '<HEART>', text)

	nbr = '[-+]?[.\d]*[\d]+[:,.\d]*'
	assert re.sub(nbr, "<NUMBER>", "-2.587 1,100,433.0") == '<NUMBER> <NUMBER>'
	text = re.sub(nbr, '<NUMBER>', text)

	# Need to do more work to split hashtag text in to tokens as in original ruby code
	hsh = (r'#(\S+)', r'<HASHTAG> \1')
	assert re.sub(hsh[0], hsh[1], '#parents #suck') == ('%s parents %s suck' % tuple(['<HASHTAG>']*2))
	text = re.sub(hsh[0], hsh[1], text)

	repunc = (r'([.!?]){2,}', r'\1 <REPEAT>')
	assert re.sub(repunc[0], repunc[1], 'ohmg!!!') == ('ohmg! <REPEAT>')
	assert re.sub(repunc[0], repunc[1], 'tonight...') == ('tonight. <REPEAT>')
	text = re.sub(repunc[0], repunc[1], text)

	elong = (r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <ELONG>')
	assert re.sub(elong[0], elong[1], 'wayyy') == ('way <ELONG>')
	text = re.sub(elong[0], elong[1], text)


	def capsub(matchobj):
		#pdb.set_trace()
		return matchobj.group(0).lower() + ' <ALLCAPS> '

	if False:
		# skipping for now because this was identifying the tags as all caps
		#caps = (r"([^a-z0-9()<>'`\-]){2,}", capsub)
		#caps = (r"[^<]*([A-Z\s']{4,})[^>]*", capsub)
		#caps = (r"\b([A-Z\s']){4,}\b", capsub)
		print re.sub(caps[0], caps[1], "I can't believe it. I WON'T believe it")
		print re.sub(caps[0], caps[1], "I can't believe it! <REPEAT>")
		assert re.sub(caps[0], caps[1], "OMG I can't believe it") == ("omg i  <ALLCAPS> can't believe it")
		assert re.sub(caps[0], caps[1], "I can't believe it") == ("I can't believe it")
		assert re.sub(caps[0], caps[1], "I can't believe it! <REPEAT>") == ("I can't believe it! <REPEAT>")
		assert re.sub(caps[0], caps[1], "I can't believe it. I WON'T believe it") == ("I can't believe it. i won't  <ALLCAPS> believe it")
		text = re.sub(caps[0], caps[1], text)

	return text
