from numpy import *
import itertools
import time
import sys
from matplotlib.pyplot import *
matplotlib.rcParams['savefig.dpi'] = 100
import pandas as pd

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        self.bptt = bptt
        self.alpha = alpha


        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        self.sparams.L = random.normal(scale=.1**0.5, size=L0.shape)
        self.params.U = random.normal(scale=.1**0.5, size=self.params.U.shape)

        # Initialize H matrix, as with W and U in part 1
        self.params.H = random_weight_matrix(*self.params.H.shape)

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

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

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####
        # param shapes
        # H = Dh x Dh
        # L = V x Dh
        # U = V x Dh

        # onehot rows of actual results
        #Y = zeros((ns, self.vdim))

        # error of output layer
        # dim T x V
        #err_o = zeros(Y.shape)
        err_os = zeros((ns, self.vdim))

        ##
        # Forward propagation
        # TRYING AS A LOOP 1st MAY WANT TO DO MATRIX MATH AFTER

        for t in range(ns):
            # shape T x Dh
            hs[t] = sigmoid(self.params.H.dot(hs[t-1]) + self.sparams.L[xs[t]])
            # shape T x V
            ps[t] = softmax(self.params.U.dot(hs[t]))
            
            # output layer error = Yhat - Y
            err_os[t] = ps[t] - make_onehot(ys[t], self.vdim)



        ##
        # Backward propagation through time

        # error assigned to each hidden node
        # dim T x Dh
        # err_hs[t+1]=0
        err_hs = zeros((ns+1, self.hdim))

        for t in reversed(range(max(ns-self.bptt,0),ns)):
            err_hs[t] += (self.params.U.T.dot(err_os[t]) +
                self.params.H.T.dot(err_hs[t+1])) * hs[t] * (1-hs[t])

            # dim V x Dh
            self.grads.U += outer(err_os[t],hs[t])
            
            # dim Dh
            self.sgrads.L[xs[t]] = err_hs[t]

            #dim Dh x Dh
            self.grads.H += outer(err_hs[t], hs[t-1])

        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        # shape T x Dh
        hs = zeros((ns+1, self.hdim))
        # predicted probas, shape T x V
        ps = zeros((ns, self.vdim))

        for t in range(ns):
            # shape Dh
            hs[t] = sigmoid(self.params.H.dot(hs[t-1]) + self.sparams.L[xs[t]])
            # shape T x V
            ps[t] = softmax(self.params.U.dot(hs[t]))
            J += -log(ps[t,ys[t]])


        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init] # emitted sequence

        #### YOUR CODE HERE ####
        h = zeros(self.hdim)
        t = 1
        while t < maxlen:
            # shape Dh
            h = sigmoid(self.params.H.dot(h) + self.sparams.L[ys[t-1]])
            # shape V
            p = softmax(self.params.U.dot(h))
            ys += [multinomial_sample(p)]
            J += -log(p[ys[t]])
            if ys[t] == end:
                break
            t += 1


        #### YOUR CODE HERE ####
        return ys, J



class ExtraCreditRNNLM(RNNLM):
    """
    Implements an improved RNN language model,
    for better speed and/or performance.

    We're not going to place any constraints on you
    for this part, but we do recommend that you still
    use the starter code (NNBase) framework that
    you've been using for the NER and RNNLM models.
    """

    def __init__(self, *args, **kwargs):
        #### YOUR CODE HERE ####
        raise NotImplementedError("__init__() not yet implemented.")
        #### END YOUR CODE ####

    def _acc_grads(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("_acc_grads() not yet implemented.")
        #### END YOUR CODE ####

    def compute_seq_loss(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("compute_seq_loss() not yet implemented.")
        #### END YOUR CODE ####

    def generate_sequence(self, init, end, maxlen=100):
        #### YOUR CODE HERE ####
        raise NotImplementedError("generate_sequence() not yet implemented.")



class RNNLMWrapper():
    """" Wraps RNNLM to save input parameters, training output, and f1 score"""

    def __init__(self, vocabsize, hidden_nodes=100, alpha=0.005,
        rseed=10, bptt=1, costevery=500):

        self.params = {}

        self.params['hidden_nodes'] = hidden_nodes
        self.params['bptt'] = bptt
        self.params['alpha'] = alpha
        self.params['anneal'] = hasattr(self.params['alpha'], '__iter__')
        L0 = zeros((vocabsize, hidden_nodes))

        if self.params['anneal']:
            alpha = alpha[0]

        self.model = RNNLM(L0, bptt=bptt, alpha=alpha, rseed=rseed)

        

        # self.params = {'hidden_nodes':hidden_nodes, 'reg':reg, 'alpha':alpha, 'k':k, 'batches':batches}

        self.costevery = costevery

        self.trainingcurve = None
        self.score = None

    def train_sgd(self, X, y, printevery=100, costevery=None, anneal_alpha=None):
        if costevery is None:
            costevery = self.costevery

        if not anneal_alpha is None:
            self.model.alpha = anneal_alpha.pop(0)
        elif self.params['anneal']:
            anneal_alpha = self.params['alpha'][:]
            self.model.alpha = anneal_alpha.pop(0)

        idxiter = range(len(y))#random_sched(batches*k, len(y), k=k)
        random.shuffle(idxiter)

        trainingcurve = self.model.train_sgd(X=X, y=y, idxiter=idxiter,
            printevery=printevery, costevery=costevery)

        if self.trainingcurve is None:
            self.trainingcurve = trainingcurve
        else:
            # update training curve
            self.trainingcurve += [(tpl[0]+self.trainingcurve[-1][0],tpl[1]) for tpl in trainingcurve[1:]]

        if not (anneal_alpha is None or anneal_alpha == []):
            self.train_sgd(X=X, y=y, printevery=printevery, costevery=costevery, anneal_alpha=anneal_alpha)

        


def plot_rnnlm_wrapper_list(rnnlm_list, label_params=None, save_name='default', ymax=None):
    ##
    # Plot comparison of learning rates here
    
    # rnnlm_list is a list of RNNLMWrapper objects
    #   this function will plot the training curves of the list
    # label_params is a list of strings that indicates which
    #   params to label

    figure(figsize=(6,4))
    cm = get_cmap('gist_rainbow')
    nlines = len(rnnlm_list)

    i = 0
    for rnnlm in rnnlm_list:
        counts, costs = zip(*rnnlm.trainingcurve)
        color = cm(1.*i/nlines)

        if label_params is None:
            label = str(i)
        else:
            label = ''
            for param in label_params:
                label += '%s=%s' % (param, rnnlm.params[param])

        plot(5*array(counts), costs, color=color, marker='o', linestyle='-', label=label)
        i += 1
    
    title(r"Learning Curve")
    xlabel("SGD Iterations"); ylabel(r"Average $J(\theta)$"); 
    ylim(ymin=0, ymax=max(1.1*max(costs),3*min(costs)) if ymax is None else ymax);
    legend()

    # Don't change this filename
    savefig("rnnlm.learningcurve.%s.png" % save_name)


def eval_wrapper(rrnlm_wrapper, vocab, vocabsize, fraction_lost, X_test, y_true, tagnames, verbose=True):
    ## Evaluate cross-entropy loss on the dev set,
    ## then convert to perplexity for your writeup
    dev_loss = rrnlm_wrapper.model.compute_mean_loss(X_test, y_true)

    ## DO NOT CHANGE THIS CELL ##
    # Report your numbers, after computing dev_loss above.
    def adjust_loss(loss, funk, q, mode='basic'):
        if mode == 'basic':
            # remove freebies only: score if had no UUUNKKK
            return (loss + funk*log(funk))/(1 - funk)
        else:
            # remove freebies, replace with best prediction on remaining
            return loss + funk*log(funk) - funk*log(q)
    # q = best unigram frequency from omitted vocab
    # this is the best expected loss out of that set
    q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
    retval = (exp(dev_loss), exp(adjust_loss(dev_loss, fraction_lost, q)))
    if verbose:
        print "Unadjusted: %.03f" % retval[0]
        print "Adjusted for missing vocab: %.03f" % retval[1]

    return(retval)


def wrapper_param_search(X, y, vocabsize, hidden_nodes=100, alpha=0.005,
        rseed=10, bptt=1, printevery=100, costevery=None,
        verbose=True):
    
    clf_list = []

    if not hasattr(hidden_nodes, "__iter__"):
        hidden_nodes = [hidden_nodes]

    if not hasattr(bptt, "__iter__"):
        bptt = [bptt]

    if not hasattr(alpha, "__iter__"):
        alpha = [alpha]

    for bptti in bptt:
        for nodei in hidden_nodes:
            for alphai in alpha:
                if verbose:
                    print('%s=%s\n%s=%s\n%s=%s' % ('bptt',bptti,'nodes',nodei,'alpha',alphai))
                modeli = RNNLMWrapper(vocabsize, hidden_nodes=nodei, alpha=alphai,rseed=rseed, bptt=bptti)
                modeli.train_sgd(X,y,printevery=printevery, costevery=costevery)
                clf_list += [modeli]

    return(clf_list)
    

def table_of_last_iter(rrnlm_list, X_test=None, y_true=None, tagnames=None, vocab=None, vocabsize=None, fraction_lost=None, verbose=True, sortby=None, write_path='search.csv'):
    include_eval = not (X_test is None or y_true is None or vocab is None or vocabsize is None or fraction_lost is None)
    
    if sortby is None:
        sortby = 'adjusted' if include_eval else 'last_score'
    
    last_iter = []
    for item in rrnlm_list:
        add = {'alpha':item.params['alpha'],
                      'bptt':item.params['bptt'],
                      'hidden_nodes':item.params['hidden_nodes'],
                      'last_score':item.trainingcurve[-1][1]}
        if include_eval:
            unadj, adj = eval_wrapper(item, vocab=vocab, vocabsize=vocabsize, fraction_lost=fraction_lost, X_test=X_test, y_true=y_true, tagnames=tagnames, verbose=verbose)
            add['unadjusted'] = unadj
            add['adjusted'] = adj
        last_iter += [add]

    df = pd.DataFrame(last_iter).sort(columns=sortby)

    if not write_path is None:
        df.to_csv(write_path)

    return(df)
