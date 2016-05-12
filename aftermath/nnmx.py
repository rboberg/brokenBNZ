from numpy import *
import pdb
import itertools
import time
import sys
from matplotlib.pyplot import *
matplotlib.rcParams['savefig.dpi'] = 100
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class NNMX(NNBase):
    """
    Implements an NN model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        wdim : word vector dimensions
        hdim : hidden layer dimensions
        odim : output layer dimensinos
        alpha : default learning rate
        rho : regularization constant
        rseed : random seed
        drop_p : probability of dropout

        bptt : number of backprop timesteps
    """

    def __init__(self, wdim, hdim=None, odim=2,
                 alpha=0.005, rho=1e-4, rseed=10, bptt=1,
                 drop_p=0.5):
        np.random.seed(rseed)
        self.rseed = rseed

        self.wdim = wdim
        #self.wdim = L.shape[1] # word vector dimensions
        #self.vdim = L.shape[0] # vocab size
        if hdim is None: hdim = self.wdim
        self.hdim = hdim
        self.odim = odim

        param_dims = dict(W1 = (self.hdim, self.wdim),
                          b1 = (self.hdim, ),
                          Ws = (self.odim, self.hdim),
                          bs = (self.odim, ))

        # word embeddings are not updated
        # no longer needed because passing word vectors X
        #self.L = L

        # no sparse updates in this model
        #param_dims_sparse = dict(L = L0.shape)
        #NNBase.__init__(self, param_dims, param_dims_sparse)
        NNBase.__init__(self, param_dims)

        #### YOUR CODE HERE ####
        # not recursive yet, but leaving bptt anyway
        self.bptt = bptt
        self.alpha = alpha
        self.rho = rho
        self.drop_p = drop_p # probability of dropping word embedding element from training


        # Initialize weight matrices
        #self.params.W1 = random.normal(scale=.1**0.5, size=self.params.W1.shape)
        #self.params.Ws = random.normal(scale=.1**0.5, size=self.params.Ws.shape)
        self.params.W1 = random_weight_matrix(*self.params.W1.shape)
        self.params.Ws = random_weight_matrix(*self.params.Ws.shape)

        # initialize bias vectors
        self.params.b1 = zeros((self.hdim))
        self.params.bs = zeros((self.odim))

        #### END YOUR CODE ####


    def _acc_grads(self, X, y):
        """
        Accumulate gradients, given a pair of training sequences:
        X = input word vectors (N x WvecDim matrix)
        y = document classifcation (as an integer)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H, U)
                and self.sgrads (for L)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect X as a matrix of word vectors
        # ith row is a word embedding for the ith word
        ns = X.shape[0]
        y = array(y).astype(int)

        #### YOUR CODE HERE ####

        # FORWARD PROP

        # X.shape = (ns, Dw)
        # no longer neded because passing word vectors X
        #X = self.L[xs,:]

        Z = (self.params.W1.dot(X.T)).T + self.params.b1
        # A.shape = (ns, Dh)
        A = sigmoid(Z)
        assert A.shape == (ns, self.hdim)

        # if dropout set A
        if self.drop_p > 0.:
            A[random.rand(*A.shape) <= self.drop_p] = 0.

        # Max each node of A over time (max of each column over all rows)
        # use argmax for use in backprop
        mx = argmax(A,0)

        # Max pooling vector
        # this will select max elements of A:
        # h.shape == (Dh,)
        h = A[mx,range(len(mx))]
        assert h.shape == (self.hdim,)

        # prediction probabilities
        ps = softmax(self.params.Ws.dot(h) + self.params.bs)

        # BACK PROP
        err_o = ps
        err_o[y] += -1

        self.grads.Ws += outer(err_o, h)
        self.grads.bs += err_o


        err_h_max = self.params.Ws.T.dot(err_o) * h * (1-h)
        assert err_h_max.shape == (self.hdim,)

        # the x vector that went in to the argmax instance of each node
        X_max = X[mx,:]
        assert X_max.shape == (self.hdim, self.wdim)

        # How to multiply by the same thing in each row (columnwise multiplication)
        # zeros((10,5)) * reshape(range(10), (10,))[:,newaxis]
        self.grads.W1 += X_max * err_h_max[:,newaxis]
        self.grads.b1 += err_h_max

        self.grads.W1 += self.rho * self.params.W1
        self.grads.Ws += self.rho * self.params.Ws

        #### END YOUR CODE ####



    def grad_check(self, X, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        # if not recurssive yet this setting of bptt does not matter
        bptt_old = self.bptt
        # single example
        if isinstance(X, ndarray): X = [X]

        for i in range(len(X)):
            self.bptt = X[i].shape[0]
            print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
            NNBase.grad_check(self, X[i], y[i], outfd=outfd, **kwargs)

        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def predict_seq_proba(self, X):
        #### YOUR CODE HERE ####
        # Expect xs as list of indices
        #ns = len(xs)
        ns = X.shape[0]

        # X.shape = (ns, Dw)
        #X = self.L[xs,:]

        Z = (self.params.W1.dot(X.T)).T + self.params.b1
        # A.shape = (ns, Dh)
        A = sigmoid(Z)
        assert A.shape == (ns, self.hdim)

        if self.drop_p > 0.:
            A = A * (1-self.drop_p)

        # Max each node of A over time (max of each column over all rows)
        # use argmax for use in backprop
        mx = argmax(A,0)

        # Max pooling vector
        # this will select max elements of A:
        # h.shape == (Dh,)
        h = A[mx,range(len(mx))]
        assert h.shape == (self.hdim,)

        # prediction probabilities
        ps = softmax(self.params.Ws.dot(h) + self.params.bs)

        return(ps)

    def predict_proba(self, X):
        if isinstance(X, ndarray): # single example
            #X is a single matrix
            return self.predict_seq_proba(X)
        else: # multiple examples / X is a list of matrices
            return [self.predict_seq_proba(xs) for xs in X]

    def predict_seq(self, X):
        return np.argmax(self.predict_seq_proba(X))

    def predict(self, X):
        if isinstance(X, ndarray): # single example
            #X is a single matrix
            return self.predict_seq(X)
        else: # multiple examples / X is a list of matrices
            return [self.predict_seq(xs) for xs in X]

    def compute_seq_loss(self, X, y):
        """
        Compute the total cross-entropy loss
        for an input matrix X and label y.

        You should run the NN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        #J = 0
        y = array(y).astype(int)

        #J = -log(ps[y])

        ps = self.predict_seq_proba(X)
        J = -log(ps[y]) + (self.rho / 2) * (sum(self.params.W1**2) + sum(self.params.Ws**2))

        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """

        if isinstance(X, ndarray): # single example
            #X is a single matrix
            return self.compute_seq_loss(X, Y)
        else: # multiple examples / X is a list of matrices
            return sum([self.compute_seq_loss(xs,y)
                       for xs,y in itertools.izip(X, Y)])

    def string_name(self):
        """
        Based on parameters of the model, generate a string name
        To be used for saving the model
        """
        return(
            (
                'model=nnmx_wdim={wdim}_hdim={hdim}_odim={odim}_alpha={alpha}' +
                '_rho={rho}_dropp={drop_p}_rseed={rseed}'
            ).format(
                wdim=self.wdim, hdim=self.hdim, odim=self.odim, alpha=self.alpha,
                rho=self.rho, drop_p=self.drop_p, rseed=self.rseed)
            )

    def compute_display_loss(self, X, y):
        """
        Optional alternative loss function for printing or diagnostics.
        """
        return roc_auc_score(y, np.array(self.predict_proba(X))[:,1])


if __name__ == '__main__':

    from datautil import load_raop_data
    from transformers import ExtractBody, PrepAndVectorize
    from sklearn.pipeline import Pipeline
    all_train_df, all_train_labels, submit_df = load_raop_data()

    wdim = 50
    hdim = 10
    # need to make sure dropout equal to 0 for
    # gradient check, otherwise will not work
    # because it is stochastic
    drop_p = 0.

    X = Pipeline([
        ('body', ExtractBody()),
        ('vec', PrepAndVectorize(d=wdim))
        ]).fit_transform(all_train_df[:4],y=1)

    y = all_train_labels[:4]

    nnmx = NNMX(wdim=wdim,hdim=hdim,drop_p=drop_p)

    print "Numerical gradient check..."
    nnmx.grad_check(X, y)




