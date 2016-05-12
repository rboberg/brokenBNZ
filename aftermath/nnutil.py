import os
import re
import regex
import cPickle as pickle
import pdb
import numpy as np
import pandas as pd
import time
from IPython.display import display


def open_nn_results(like, path='results', exactmatch=False, verbose=True):
    """
    Open a list of nn results with file names matching  like
    """
    files = os.listdir(path)
    fmatch = [f for f in files if not re.match(like, f) is None]
    if len(fmatch) == 0:
        fmatch = [f for f in files if not re.match('.*'+like+'.*', f) is None]
    if exactmatch:
        raise NotImplementedError("exactmatch not implemented")

    assert len(fmatch) != 0, "Could not find any files matching the pattern %s" % like

    nn_results = [open_nn_result(f, path, verbose=verbose) for f in fmatch]
    return(nn_results)


def open_nn_result(f, path, verbose=False):
    """
    Open a single nn result
    """
    #TODO configure to accept directories
    if verbose: print "loading %s" % f
    if os.path.isdir(path+'/'+f):
        return {f:[open_nn_result(f+'/'+fi, path, verbose) for fi in os.listdir(path+'/'+f)]}
    with open(path+'/'+f, 'rb') as fid:
        qcmbr = pickle.load(fid)
    return qcmbr

def open_costs(like, path='results', exactmatch=False, verbose=True):
    """
    Open a list of cost results
    """
    nn_results = open_nn_results(
        like=like, path=path,
        exactmatch=exactmatch, verbose=verbose)

    if 'costs' in nn_results[0]:
        # nn_results assumed to be a list of file results
        return [{like:[nn['costs'] for nn in nn_results]}]
    else:
        # nn_results assumed to be a list of dicts of file results
        return [{k:[vi['costs'] for vi in v] for k,v in nn.iteritems()} for nn in nn_results]




def cost_iter_summary(costs=None, like=None, asDF = True, path='results', exactmatch=False, verbose=True):
    """
    take a list of cost results (as returned by open_costs) or regex pattern (like) to look up costs
    and get summary statistics (mean, median, sd) of the various results by training
    iteration counts.
    """

    assert (costs is None) and (not like is None), "either costs or like must be provided"
    if costs is None:
        costs = open_costs(like, path=path, exactmatch=exactmatch, verbose=verbose)

    d = []
    for cost_list in costs:
        for k,v in cost_list.iteritems():
            d += [({
                'id':k,
                'count':count[0][0],
                'median':np.median(np.array([tup[1][0] for tup in count])),
                'mean':np.mean(np.array([tup[1][0] for tup in count])),
                'std':np.mean(np.std([tup[1][0] for tup in count])),
                'n': len(v)
                }) for count in zip(*v)]

    return pd.DataFrame(d) if asDF else d

def cost_iter_compare(likes, asDF=True, path='results', exactmatch=False,
    verbose=True, opts2col=True, drop_all_same=True):
    """
    take a list of regex patterns (like) and return NN results
        opts2col: will call parse_opts_to_column on result if asDF = True
    """
    costs = []
    for like in likes:
        costs += cost_iter_summary(like=like, asDF=False, path=path,
            exactmatch=exactmatch, verbose=verbose)

    if asDF:
        costs = pd.DataFrame(costs)
        if opts2col:
            costs = parse_opts_to_column(costs, idcol='id', drop_all_same=drop_all_same)

    return costs

def drop_all_same_columns(df):
    """
    Will drop any new columns where all the values are identical for every row
    """
    for col in df.columns.values:
        if len(pd.unique(df[col])) == 1:
            df = df.drop(col,1)
    return(df)

def ppdf(df, allrows=True, allcols=True, allvals=False, ipy=True):
    """
    Pretty print a data frame
        allrows: will show all rows if True
        allvals: will show the entire value (untruncated) for each cell if True
        allcols: will show all cols if True
    """

    opts = []

    if allrows: opts += ['display.max_rows', len(df)]
    if allcols: opts += ['display.max_columns', df.shape[1]]
    if allvals: opts += ['display.max_colwidth', -1]

    with pd.option_context(*opts):
        if ipy:
            display(df)
        else:
            print(df)

def parse_opts_to_column(results, idcol='id', drop_all_same=True):
    """
    Take a dataframe of results, as returned by cost_iter_compare with asDF = True
    Give an id column with the format arg1=val1_arg2=val2_...
    It will add a column named arg1 with value val1, arg2 with value val2, ...
    drop_all_same = True, will drop any new columns where
        all the values are identical for every row
    """
    fields = []
    for field_str in results[idcol]:
        x = regex.match(r'(_*([^=]+)=([^_]+)($|_))+',field_str)
        fields += [{i[0]:i[1] for i in zip(x.captures(2), x.captures(3))}]
    df = pd.DataFrame(fields, results.index)
    drop_all_same = True
    if drop_all_same: df = drop_all_same_columns(df)

    return pd.concat((results,df), axis=1)

def save_experiment(results, filename, description, path='experiments'):
    cucumber = (description, results, time.strftime('%Y-%m-%d_%H:%M:%S'))
    fstring = '{p}/{f}_{t}.bin'.format(
        p=path, f=filename, t=time.strftime('%Y%m%d')
        )
    with open(fstring, 'wb') as fid:
        pickle.dump(cucumber, fid, protocol=pickle.HIGHEST_PROTOCOL)

def save_all_results(resultpath='results', savepath='result_summary'):
    likes = ['.*']
    results = cost_iter_compare(likes=['.*'],
        path=resultpath, verbose=False).sort(['count', 'median'])
    save_experiment(results=results, filename='all_results',
        description='Saving all results from directory "{f}"'.format(f=resultpath),
        path=savepath)

def list_experiments(path='experiments', results=False):
    files = os.listdir(path)
    for file in files:
        with open(path + '/' + file, 'rb') as fid:
            qcmbr = pickle.load(fid)
            print '*********\n{file}\n---\n{t}\n---\n{des}'.format(
                file=file, t=qcmbr[2], des=qcmbr[0])
            if results:
                print '---'
                print qcmbr[1]
            print '*********'

### DEPRECATED

def open_nn_results_old_(like, path='results', exactmatch=False, verbose=True):
    """
    Open a list of nn results with file names matching  like
    """
    files = os.listdir(path)
    fmatch = [f for f in files if not re.match(like, f) is None]
    if len(fmatch) == 0:
        fmatch = [f for f in files if not re.match('.*'+like+'.*', f) is None]
    if exactmatch:
        raise NotImplementedError("exactmatch not implemented")

    assert len(fmatch) != 0, "Could not find any files matching the pattern %s" % like



    nn_results = [open_nn_result_old_(f, path, verbose=verbose) for f in fmatch]
    return(nn_results)


def open_nn_result_old_(f, path, verbose=False):
    """
    Open a single nn result
    """
    if verbose: print "loading %s" % f
    with open(path + '/' + f, 'rb') as fid:
        qcmbr = pickle.load(fid)
    return qcmbr

def open_costs_old_(like, path='results', exactmatch=False, verbose=True):
    """
    Open a list of cost results
    """
    nn_results = open_nn_results_old_(
        like=like, path=path,
        exactmatch=exactmatch, verbose=verbose)

    return [nn['costs'] for nn in nn_results]
