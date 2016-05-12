"""
Tokenizers
"""
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# Lastly, some classes to handle string tokenizing that we will use in multiple sections:
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class SnowballStemTokenizer(object):
    def __init__(self):
        self.stmr = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.stmr.stem(t) for t in word_tokenize(doc)]
    
class PorterStemTokenizer(object):
    def __init__(self):
        self.stmr = PorterStemmer()
    def __call__(self, doc):
        return [self.stmr.stem(t) for t in word_tokenize(doc)]

class PuncTokenizer(object):
    def __init__(self):
        self.reg = RegexpTokenizer(r'[\s\.\,\:\-\;\(\)\[\]\{\}\!\?]+',gaps=True)
    def __call__(self, doc):
        return self.reg.tokenize(doc)
    
class SpaceTokenizer(object):
    def __init__(self):
        self.tknzr = WhitespaceTokenizer()
    def __call__(self, doc):
        return [t for t in self.tknzr.tokenize(doc)]

class PTBTokenizer(object):
    def __init__(self):
        self.tknzr = TreebankWordTokenizer()
    def __call__(self, doc):
        return [t for t in self.tknzr.tokenize(doc)]