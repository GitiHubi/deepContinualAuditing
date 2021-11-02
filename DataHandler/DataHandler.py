# import class libraries
import datetime as dt
import numpy as np
import pandas as pd
import os

# class Utilities
class DataHandler(object):

    def __init__(self):

        pass

    def get_Philadelphia_data(self, experiment_parameters, task_id):

        # get dataset parameters of the experiment
        data_dir = experiment_parameters['data_dir']

        # read the non-encoded transactional data
        transactions = pd.read_csv(os.path.join(data_dir, 'task_{}_data'.format(str(task_id)), 'payment_data_t{}.csv'.format(str(task_id))), sep=',', encoding='utf-8')

        # log data loading progress
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional philadelphia data of shape {} rows and {} columns successfully loaded.'.format(now, str(transactions.shape[0]), str(transactions.shape[1])))

        # read the encoded transactional data
        transactions_encoded = pd.read_csv(os.path.join(data_dir, 'task_{}_data'.format(str(task_id)), 'payment_data_t{}_encoded.csv'.format(str(task_id))), sep=',', encoding='utf-8')

        # log data loading progress
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional encoded philadelphia data of shape {} rows and {} columns successfully loaded.'.format(now, str(transactions_encoded.shape[0]), str(transactions_encoded.shape[1])))

        # determine encoded transactions ids
        transactions_encoded_ids = transactions_encoded['id']

        # remove non training relevant fields
        transactions_encoded = transactions_encoded.drop(columns=['id', 'task'], axis=1)

        # convert to numpy array of floats
        transactions_encoded = transactions_encoded.to_numpy().astype(np.float32)

        # return transactions and encoded transactions
        return transactions, transactions_encoded, transactions_encoded_ids
