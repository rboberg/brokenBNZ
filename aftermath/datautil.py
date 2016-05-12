"""
File with functions for extracting data
for the RAOP kaggle competition
"""

import numpy as np
import json as json
import pandas as pd
import csv as csv
import random as rand

# Helper methods to pull and submit data:
def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data

def make_submission_csv(predictions, ids, submission_name, path = '../predictions'):
    with open(path+'/'+submission_name+'.csv', 'w') as csvfile:
        field_names = ['request_id', 'requester_received_pizza']
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        csv_data = zip(ids, predictions)
        for row in csv_data:
            writer.writerow({field_names[0]:row[0], field_names[1]:int(row[1])})

# Helper methods for pulling columns from the dataset:
def name2index(df, names):
    return_single = False

    if type(names) == type([]):
       names = np.array(names)
    elif type(names) != type(np.array([])):
        names = np.array([names])
        return_single = True

    inds = np.where(np.in1d(df.columns, np.array(names)))[0]

    return inds[0] if return_single else inds

def load_raop_data():
    """
    returns (all_train_df, all_train_labels, submit_df)
    """

    # load data from JSON file as list of dicts
    all_train_dict_list = load_json_file('../data/train.json')
    submit_dict_list =  load_json_file('../data/test.json')

    n_all = len(all_train_dict_list)
    n_submit = len(submit_dict_list)

    # shuffle data to avoid biased splits of train / dev data
    rand.shuffle(sorted(all_train_dict_list, key=lambda k: k['request_id']))

    # process labels
    all_train_labels = np.array([x['requester_received_pizza'] for x in all_train_dict_list])

    # pandas is useful for turning dicts in to matrix-like objects
    # where each column is an numpy array
    submit_df = pd.DataFrame(submit_dict_list)
    all_train_df = pd.DataFrame(all_train_dict_list)

    # limit train to columns available in submit_df
    submit_cols = submit_df.columns
    all_train_df = all_train_df[submit_cols]

    return (all_train_df, all_train_labels, submit_df)