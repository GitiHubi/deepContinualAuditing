from torch.utils import data

# define payment dataset
class PaymentDataset(data.Dataset):

    # define the class constructor
    def __init__(self, payments_encoded, payment_ids, task_id):

        # set payments encoded
        self.payments_encoded = payments_encoded

        # set payments ids
        self.payment_ids = payment_ids

        # set task id
        self.task_id = task_id

    # define the length method
    def __len__(self):

        # returns the number of payments
        return len(self.payments_encoded)

    # define the get item method
    def __getitem__(self, index):

        # determine encoded mini batch
        payments_encoded_batch = self.payments_encoded[index, :]

        # determine id mini batch
        payments_ids_batch = self.payment_ids[index]

        # return sequences and target
        return payments_ids_batch, payments_encoded_batch