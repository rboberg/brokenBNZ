from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import pdb
import numpy as np
from datautil import name2index
from getwv import tokenvec
from glove_tknz import glove_prep
from nltk import word_tokenize
WVEC_PATH__ = '../data/twitter'
TOPDIR__ = WVEC_PATH__+'/50d'
D__ = 50
DIRDEPTH__ = 3
# for example:
#tokenvec(['a','the'],TOPDIR__,DIRDEPTH__)

"""
Contains transformers for data manipulation
"""

class ExtractColumnsTransformer(TransformerMixin):
    """
    Transformer for extracting columns by index
    """

    def __init__(self, cols=[0]):
        self.cols = cols

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        cols = self.cols
        return X[:,cols]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class ExtractColumnsByName(TransformerMixin):
    """
    Transformer for extracting columns by index
    """

    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        cols = self.cols
        return X.loc[:,cols].values

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

### Set up Numeric Activity Variables, including processor class for all activity feature analysis.
ACTIVITY_VARS__ = ['requester_account_age_in_days_at_request',
                'requester_days_since_first_post_on_raop_at_request',
                'requester_number_of_comments_at_request',
                'requester_number_of_comments_in_raop_at_request',
                'requester_number_of_posts_at_request',
                'requester_number_of_posts_on_raop_at_request',
                'requester_number_of_subreddits_at_request',
                'requester_upvotes_minus_downvotes_at_request',
                'requester_upvotes_plus_downvotes_at_request'
                ]

class ExtractActivities(ExtractColumnsByName):
    """
    Extract activity related columns
    """
    def __init__(self):
        ExtractColumnsByName.__init__(self, ACTIVITY_VARS__)




### Define quick classes that we can use to isolate the title and body columns in our data.
TITLE_COLUMN__ = ['request_title']
BODY_COLUMN__ = ['request_text_edit_aware']

class ExtractBody(ExtractColumnsByName):
    """
    Extract body text
    """
    def __init__(self):
        ExtractColumnsByName.__init__(self, BODY_COLUMN__)

class ExtractTitle(ExtractColumnsByName):
    """
    Extract title text
    """
    def __init__(self):
        ExtractColumnsByName.__init__(self, TITLE_COLUMN__)

class ExtractAllText(ExtractColumnsByName):
    """
    Extract Body and Title in Two Columns
    """
    def __init__(self):
        ExtractColumnsByName.__init__(self, np.array([TITLE_COLUMN__, BODY_COLUMN__]))

USER_NAME_COLUMN__ = ['requester_username']
class ExtractUser(ExtractColumnsByName):
    def __init__(self):
        ExtractColumnsByName.__init__(self, USER_NAME_COLUMN__)


class ConcatStringTransformer(TransformerMixin):
    """
    Joins several columns of strings in to a single column
    """
    def __init__(self):
        return None

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):

        if len(X.shape) == 1:
            return X
        else:
            n_feat = X.shape[1]
            new_list = []
            for i in range(X.shape[0]):
                new_list.append('.'.join(X[i,:]))

            return np.array(new_list)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class DesparseTransformer(TransformerMixin):
    """
    Change a sparse matrix to a dense one
    """
    def __init__(self):
        return None

    def transform(self, X, **transform_params):
        if (type(X) != type(np.array(1))):
            return X.toarray()

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

def rejoin(l0,l1=None):
    """
    this function rejoins special tokens that
    are surround by angle brackets ('<TOKEN>') but get
    separated by the parser in to ['<','TOKEN','>']
    """
    while len(l0) != 0:
        if l1 is None:
            l1 = []
        if l0[0] == '<':
            if len(l0) >= 3:
                if l0[2] == '>':
                    l1 += [l0.pop(0) + l0.pop(0) + l0.pop(0)]
        else:
            l1 += [l0.pop(0)]
    return l1

class TokenizeTransformer(TransformerMixin):
    """
    Transforms a list of strings (docs) in to a list of token lists
    """
    def __init__(self, tokenizer, rejoin_angle=True):
        #self.tokenizer = np.vectorize(tokenizer)
        self.tokenizer = tokenizer
        self.rejoin_angle = rejoin_angle
        return None

    def transform(self, X, **transform_params):
        if(not hasattr(X, '__iter__')): X = [X]
        if self.rejoin_angle:
            return [rejoin(self.tokenizer(Xi[0])) for Xi in X]
        else:
            return [self.tokenizer(Xi[0]) for Xi in X]

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class TwitterPrep(TransformerMixin):
    """
    preprocesses Twitter data ala GloVe
    """
    def __init__(self):
        self.fprep = np.vectorize(glove_prep)
        return None

    def transform(self, X, **transform_params):
        return self.fprep(X)

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class WordvecTransformer(TransformerMixin):
    """
    Transforms a list of lists of tokens in to a list of lists of word vectors
    """
    def __init__(self, topdir = TOPDIR__, dirdepth = DIRDEPTH__, d = D__, missing0 = True, nowarnings=True):
        self.topdir = topdir
        self.dirdepth = dirdepth
        self.nowarnings = nowarnings
        self.d = d
        self.missing0 = missing0
        return None

    def transform(self, X, **transform_params):
        if(not hasattr(X, '__iter__')): X = [X]
        return [np.array(
                tokenvec(
                    Xi,
                    self.topdir,
                    self.dirdepth,
                    nowarnings=self.nowarnings,
                    d=self.d,
                    missing0=self.missing0
                )) for Xi in X]

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class AverageWordvec(TransformerMixin):
    """
    Takes a list of matrices of word vectors
    For each item in the list it computes the average word vector
    """
    def __init__(self):
        return None

    def transform(self, X, **transform_params):
        return np.array([np.mean(np.array(Xi), axis = 0) for Xi in X])

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class MaxPool(TransformerMixin):
    """
    Takes a list of matrices of word vectors
    For each item in the list it computes the max element word vector
    """
    def __init__(self):
        return None

    def transform(self, X, **transform_params):
        return np.array([np.array(Xi).max(axis = 0) for Xi in X])

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class MinPool(TransformerMixin):
    """
    Takes a list of matrices of word vectors
    For each item in the list it computes the min element word vector
    """
    def __init__(self):
        return None

    def transform(self, X, **transform_params):
        return np.array([np.array(Xi).min(axis = 0) for Xi in X])

    def fit(self, X, y, **fit_params):
        #do nothing
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

def PrepAndVectorize(d, wvec_path = WVEC_PATH__):
    """
    shortcut for pipeline to Prep, Tokenize, and get Word Vectors
    for a single column of text
    """

    return(Pipeline([
        ('prep', TwitterPrep()),
        ('tknzr', TokenizeTransformer(word_tokenize, rejoin_angle=True)),
        ('wordvec', WordvecTransformer(topdir=wvec_path+'/'+str(d)+'d', d=d))
        ]))

