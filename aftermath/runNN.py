import optparse
import os
import numpy as np
import time

# sklearn
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline

#### DOMESTIC IMPORTS
from datautil import load_raop_data
from nnmx import NNMX
from nnmx2 import NNMX2
from cnn2 import CNN2
from cnn1 import CNN1
from nn.base import NNBase
from nnutil import save_all_results

### Define quick classes that we can use to isolate the title and body columns in our data.
from transformers import ExtractBody, ExtractTitle, ExtractAllText, ExtractUser
from transformers import ConcatStringTransformer, DesparseTransformer, TokenizeTransformer
from transformers import TwitterPrep, WordvecTransformer, AverageWordvec, MaxPool, MinPool
from transformers import PrepAndVectorize

###
# This is for training, testing, and saving results of a NN
###


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    #parser.add_option("--test",action="store_true",dest="test",default=False)
    #parser.add_option("--plotEpochs",action="store_true",dest="plotEpochs",default=False)
    #parser.add_option("--plotWvecDim",action="store_true",dest="plotWvecDim",default=False)

    # Optimizer
    # minibatch of 0 means no minibatches, just iterate through
    parser.add_option("--minibatch",dest="minibatch",type="int",default=0)
    #parser.add_option("--optimizer",dest="optimizer",type="string",
    #    default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--printevery",dest="printevery",type="int",default=4e4)
    parser.add_option("--annealevery",dest="annealevery",type="int",default=0) # anneal every this many epochs

    parser.add_option("--alpha",dest="alpha",type="float",default=0.005)
    parser.add_option("--rho",dest="rho",type="float",default=1e-5)
    parser.add_option("--drop_p",dest="drop_p",type="float",default=0.5)

    parser.add_option("--wdim",dest="wdim",type="int",default=50)
    parser.add_option("--hdim",dest="hdim",type="int",default=200)
    parser.add_option("--odim",dest="odim",type="int",default=2)
    parser.add_option("--rseed",dest="rseed",type="int",default=207)
    parser.add_option("--context",dest="context",type="int",default=1)

    #parser.add_option("--outFile",dest="outFile",type="string",
    #    default="models/test.bin")
    #parser.add_option("--inFile",dest="inFile",type="string",
    #    default="models/test.bin")
    #parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="NNMX")

    (opts,args)=parser.parse_args(args)


    # name of folder to store results in
    resfolder =  '_'.join(
        ['{k}={v}'.format(k=k,v=v) for k,v in vars(opts).iteritems()]
        )

    resfolder += '_timestamp={t}'.format(t=time.strftime('%Y%m%d%H%M%S'))
    resfolder = 'results/'+resfolder
    print resfolder

    if not os.path.exists(resfolder):
        os.makedirs(resfolder)

    ### Set up the training and test data to work with throughout the notebook:
    np.random.seed(opts.rseed)

    all_train_df, y, submit_df = load_raop_data()

    # useful for sklearn scoring
    #roc_scorer = make_scorer(roc_auc_score)
    n_all = all_train_df.shape[0]

    # set up kFolds to be used in the rest of the project
    kf = KFold(n_all, n_folds = 10, random_state=opts.rseed)

    body_vecs = Pipeline([
        ('body', ExtractBody()),
        ('vec', PrepAndVectorize(d=opts.wdim))
        ]).fit_transform(X=all_train_df,y=1)

    for train, test in kf:
        nn = init_model(opts)
        if opts.minibatch == 0:
            idxiter = list(train)*opts.epochs
            annealevery=len(train)*opts.annealevery
            printevery=opts.printevery
        else:
            idxiter = NNBase.randomiter(
                N=opts.epochs*len(train)/opts.minibatch,
                pickfrom=train,batch=opts.minibatch)
            annealevery=len(train)*opts.annealevery/opts.minibatch
            printevery=opts.printevery/opts.minibatch

        nn.train_sgd(body_vecs, y, idxiter=idxiter,
                       devidx=test, savepath=resfolder,
                       costevery=printevery, printevery=printevery,
                       annealevery=annealevery)

    save_all_results(resultpath = 'results', savepath = 'result_summary')

def init_model(opts):
    if (opts.model == 'NNMX'):
        nn = NNMX(wdim=opts.wdim, hdim=opts.hdim, rseed=opts.rseed,
            rho=opts.rho, drop_p=opts.drop_p, alpha=opts.alpha,
            odim=opts.odim)
    elif (opts.model == 'NNMX2'):
        nn = NNMX2(wdim=opts.wdim, hdim=opts.hdim, rseed=opts.rseed,
            rho=opts.rho, drop_p=opts.drop_p, alpha=opts.alpha,
            odim=opts.odim)
    elif (opts.model == 'CNN1'):
        nn = CNN1(wdim=opts.wdim, hdim=opts.hdim, rseed=opts.rseed,
            rho=opts.rho, drop_p=opts.drop_p, alpha=opts.alpha,
            odim=opts.odim, context=opts.context)
    elif (opts.model == 'CNN2'):
        nn = CNN2(wdim=opts.wdim, hdim=opts.hdim, rseed=opts.rseed,
            rho=opts.rho, drop_p=opts.drop_p, alpha=opts.alpha,
            odim=opts.odim, context=opts.context)
    else:
        raise '%s is not a valid neural network so far only NNMX, NNMX2, CNN2'%opts.model
    return nn


if __name__=='__main__':
    run()
