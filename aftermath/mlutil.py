"""
Helper methods for Machine Learning activities
like training, testing, scoring etc.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# Helper methods that are focused on class rebalancing:
def balance_samples(y, method='oversample', rseed=207):
    """
    Reblance a sample of data
    """

    class_counts = np.bincount(y)
    np.random.seed(rseed)

    maxi = np.argmax(class_counts)
    new_idx = np.argwhere(y==maxi)

    for i in range(len(class_counts)):
        if i != maxi:
            mult = class_counts[maxi]/class_counts[i]
            rem = class_counts[maxi] - class_counts[i]*mult
            idxi = np.argwhere(y==i)
            np.random.shuffle(idxi)
            for j in range(mult):
                new_idx = np.vstack((new_idx,idxi))
            new_idx = np.vstack((new_idx,idxi[:rem]))

    np.random.shuffle(new_idx)

    return np.reshape(new_idx, (new_idx.shape[0],))

def oversample_kfold(kf, y):
    """
    oversample underweighted class in a given kfolds
    """

    kf_over = []
    for ti, di in kf:
        yt = y[ti]
        ti_over = ti[balance_samples(yt)]
        kf_over.append((ti_over, di))
    return kf_over

# Helper methods for printing and scoring:
def test_kfolds(X, y, kf, model, verbose=1, balance=False):
    """
    test kfolds
    """

    roc_auc_list = []

    for train_i, dev_i in kf:
        if balance:
            train_i_orig = train_i
            y_train = y[train_i_orig]
            train_i = train_i_orig[balance_samples(y_train)]


        X_train = X[train_i]
        X_dev = X[dev_i]

        model.fit(X_train, y[train_i])

        dev_pred = model.predict(X_dev)

        roc_auc_i = roc_auc_score(y[dev_i], dev_pred)
        roc_auc_list.append(roc_auc_i)
        if verbose > 1:
            print('ROC AUC:',roc_auc_i)

    if verbose > 0:
        print 'N: %d, Mean: %f, Median: %f, SD: %f' %(len(kf), np.mean(roc_auc_list), np.median(roc_auc_list), np.std(roc_auc_list))

    return roc_auc_list

def print_scores(scores):
    "print scores (as returned from test_kfolds)"
    print 'N: %d, Mean: %f, Median: %f, SD: %f' %(len(scores), np.mean(scores), np.median(scores), np.std(scores))

