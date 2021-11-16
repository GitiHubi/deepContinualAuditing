from torch.utils import data
import torch
import numpy as np
import pandas as pd


class PaymentDataset(data.Dataset):
    """ Implements Payment Dataset. """

    # define the class constructor
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # set payments encoded and payment_ids
        self.payments_encoded, self.payment_ids, self.payment_depts = self.get_Philadelphia_data(data_dir)

        # set targets equal to the inputs (for computing the reconstruction loss)
        self.targets = [0] * len(self.payments_encoded)

    def get_Philadelphia_data(self, data_dir):
        """ Loads dataset from a CSV file and creats two individual arrays for
            encoded payments and transaction IDs.
        """
        # read the encoded transactional data
        print("Loading data ...")
        transactions_encoded = pd.read_csv(data_dir, sep=',', encoding='utf-8')
        print("Data is loaded now.")

        # determine encoded transactions ids and depts
        transactions_encoded_ids = transactions_encoded['id']
        transactions_encoded_depts = transactions_encoded['dept']

        # remove non training relevant fields
        transactions_encoded = transactions_encoded.drop(columns=['id', 'dept'], axis=1)

        # convert to numpy array of floats
        transactions_encoded = transactions_encoded.to_numpy().astype(np.float32)
        transactions_encoded_ids = transactions_encoded_ids.to_numpy().astype(np.float32)
        transactions_encoded_depts = transactions_encoded_depts.to_numpy()#.astype(np.float32)

        # return transactions and encoded transactions
        return transactions_encoded, transactions_encoded_ids, transactions_encoded_depts

    # define the length method
    def __len__(self):

        # returns the number of payments
        return len(self.payments_encoded)

    # define the get item method
    def __getitem__(self, index):

        # determine encoded mini batch
        payments_encoded_batch = torch.tensor(self.payments_encoded[index, :])

        # determine id mini batch
        payments_ids_batch = self.payment_ids[index]
        payments_depts_batch = self.payment_depts[index]

        #           input,                    target                   id                  dept
        return payments_encoded_batch, payments_encoded_batch , payments_ids_batch, payments_depts_batch
