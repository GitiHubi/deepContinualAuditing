# import class libraries
import numpy as np
from scipy import stats
import pandas as pd
from faker import Faker


# class for generation of artificial anomalies
class AnomalyHandler(object):

    def __init__(self, dataset, categorical_attributes, seed):
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        self.dataset = dataset
        self.categorical_attributes = categorical_attributes

    def generate_global_anomalies_chicago(self, n=10):
        """
        generate global anomalies based on chicago dataset
        :param n: number of anomalies to generate
        :return: pandas DataFrame with generated global anomalies
        """
        def get_voucher_number():
            return 'PV' + str(np.random.randint(low=100000, high=999999))

        def get_amount():
            return np.random.rand() * np.random.randint(low=100000, high=999999)

        def get_check_date():
            d = str(np.random.randint(low=1, high=28))
            m = str(np.random.randint(low=1, high=12))
            return d + '/' + m + '/2019'

        def get_department_name():
            return 'DEPARTMENT OF ' + self.fake.job()

        def get_contract_number():
            return str(np.random.randint(low=10000, high=99999))

        def get_vendor_name():
            return self.fake.company() + ' ' + self.fake.company_suffix()

        def get_cashed():
            return self.fake.boolean()

        def get_sign():
            return np.random.choice(['POS', 'NEG'])

        np.random.seed(self.seed)
        global_anomalies = []
        for i in range(n):
            sample = {'VOUCHER NUMBER': get_voucher_number(),
                      'AMOUNT': get_amount(),
                      'CHECK DATE': get_check_date(),
                      'DEPARTMENT NAME': get_department_name(),
                      'CONTRACT NUMBER': get_contract_number(),
                      'VENDOR NAME': get_vendor_name(),
                      'CASHED': get_cashed(),
                      'SIGN': get_sign(),
                      'TYPE': 'global',
                      'CLASS': 1}

            global_anomalies.append(sample)

        global_anomalies = pd.DataFrame(global_anomalies)

        # check if global anomalies are not in original dataset
        for i, global_anomaly in global_anomalies.iterrows():
            df_tmp = self.dataset[(self.dataset['VENDOR NAME'] == global_anomaly['VENDOR NAME']) &
                                  (self.dataset['DEPARTMENT NAME'] == global_anomaly['DEPARTMENT NAME'])]
            if not df_tmp.empty:
                raise Exception('global outliers need to be executed with different seed')

        return global_anomalies

    def generate_local_anomalies_chicago(self, n=10, top_frequent_values=10):
        """
        generate local anomalies based on chicago dataset
        :param n: number of anomalies to generate
        :param top_frequent_values: attribute's top N most frequent values to use for sampling
        :return: pandas DataFrame with generated local anomalies
        """
        def get_amount():
            return np.random.rand() * np.random.randint(low=1, high=999999)

        np.random.seed(self.seed)
        local_anomalies = []
        i = 0
        while len(local_anomalies) < n:
            sample = {}
            for cat_attr in self.categorical_attributes:
                # select top most frequent values of this attribute
                freq_values = self.dataset[cat_attr].value_counts().nlargest(top_frequent_values).index.tolist()
                # sample from the most frequent values
                sample[cat_attr] = np.random.choice(freq_values)

            # add amount manually
            sample['AMOUNT'] = get_amount()
            sample['TYPE'] = 'local'
            sample['CLASS'] = 1

            # append generated anomaly only when it is not in the original data
            if self.dataset[(self.dataset['VENDOR NAME'] == sample['VENDOR NAME']) &
                            (self.dataset['DEPARTMENT NAME'] == sample['DEPARTMENT NAME'])].empty:
                local_anomalies.append(sample)
            else:
                i += 1
                # print a warning if there are too many generated anomalies being rejected
                if (i % 1000 == 0) and (i != 0):
                    print('{} generated local anomalies were rejected -> consider another seed or increase top '
                          'frequent values parameter'.format(i))

        local_anomalies = pd.DataFrame(local_anomalies)

        return local_anomalies

    # generate global anomalies based on philadelphia dataset
    def generate_global_anomalies_philly(self, n=10):
        """
        generate global anomalies based on philadelphia dataset
        :param n: number of anomalies to generate
        :return: pandas DataFrame with generated global anomalies
        """
        def get_sub_obj():
            return str(np.random.randint(low=800, high=999))

        def get_document_no():
            return str('PHIL') + str(np.random.randint(low=0, high=99999999)).zfill(8)

        def get_dept():
            return np.random.randint(low=75, high=99)

        def get_char():
            return np.random.choice([1, 8, 9])

        def get_fm():
            return np.random.choice([13, 14, 15, 16])

        def get_vendor_name():
            return self.fake.company() + ' ' + self.fake.company_suffix()

        def get_doc_ref_no_prefix():
            return ''.join([self.fake.random_uppercase_letter() for _ in range(4)])

        def get_contract_number():
            return ''.join([self.fake.random_uppercase_letter() for _ in range(2)]) + str(np.random.randint(low=1, high=9999)).zfill(4)

        def get_amount():
            return np.random.choice([np.random.randint(low=-100000, high=-5000) - np.random.random(), np.random.randint(low=1000000, high=99999999) + np.random.random()])

        # init random seed
        np.random.seed(self.seed)

        # init global anomalies
        global_anomalies = []

        # iterate over number of samples
        for i in range(n):

            # sample single global anomaly
            sample = {'sub_obj': get_sub_obj(),
                      'document_no': get_document_no(),
                      'dept': get_dept(),
                      'char_': get_char(),
                      'fm': get_fm(),
                      'vendor_name': get_vendor_name(),
                      'doc_ref_no_prefix': get_doc_ref_no_prefix(),
                      'contract_number': get_contract_number(),
                      'transaction_amount': get_amount(),
                      'TYPE': 'global',
                      'CLASS': 1}

            # collect sampled random global anomaly
            global_anomalies.append(sample)

        # convert all global anomalies to pandas data frame
        global_anomalies = pd.DataFrame(global_anomalies)

        # check if global anomalies are not in original dataset
        for i, global_anomaly in global_anomalies.iterrows():

            # check for potential duplicates
            df_tmp = self.dataset[(self.dataset['document_no'] == global_anomaly['document_no'])
                                  & (self.dataset['sub_obj'] == global_anomaly['sub_obj'])
                                  & (self.dataset['fm'] == global_anomaly['fm'])
                                  & (self.dataset['dept'] == global_anomaly['dept'])
                                  & (self.dataset['doc_ref_no_prefix'] == global_anomaly['doc_ref_no_prefix'])]

            # case: no similar transactions found in original dataset
            if not df_tmp.empty:

                # raise exception
                raise Exception('global outliers need to be executed with different seed')

        # return created global anomalies
        return global_anomalies

    # generate local anomalies based on philadelphia dataset
    def generate_local_anomalies_philly(self, n=10, top_frequent_values=10):
        """
        generate local anomalies based on philadelphia dataset
        :param n: number of anomalies to generate
        :param top_frequent_values: attribute's top N most frequent values to use for sampling
        :return: pandas DataFrame with generated local anomalies
        """
        def get_amount(amount_mean, amount_std):
            return abs(np.random.normal(amount_mean, amount_std, 1))[0]

        # determine amount mean and variance
        amount_mean = stats.mode(self.dataset['transaction_amount'])[0]
        amount_std = np.std(self.dataset['transaction_amount']) / 400.0

        # init random seed
        np.random.seed(self.seed)

        # init local anomalies
        local_anomalies = []

        # init local anomaly count
        i = 0

        # iterate over number of local anomalies
        while len(local_anomalies) < n:

            # init single local anomaly
            sample = {}

            # iterate over categorical attribute
            for cat_attr in self.categorical_attributes:

                # select top most frequent values of this attribute
                freq_values = self.dataset[cat_attr].value_counts().nlargest(top_frequent_values).index.tolist()

                # sample from the most frequent values
                sample[cat_attr] = np.random.choice(freq_values)

            # add amount manually
            sample['transaction_amount'] = get_amount(amount_mean=amount_mean, amount_std=amount_std)
            sample['TYPE'] = 'local'
            sample['CLASS'] = 1

            # case: generated local anomaly is not in regular data
            if self.dataset[(self.dataset['document_no'] == sample['document_no'])
                            & (self.dataset['sub_obj'] == sample['sub_obj'])
                            & (self.dataset['fm'] == sample['fm'])
                            & (self.dataset['dept'] == sample['dept'])
                            & (self.dataset['doc_ref_no_prefix'] == sample['doc_ref_no_prefix'])
                            & (self.dataset['vendor_name'] == sample['vendor_name'])].empty:

                # append local anomaly
                local_anomalies.append(sample)

            # case: generated local anomaly is in regular data
            else:

                # increase total count of to be generated anomalies
                i += 1

                # print a warning if there are too many generated anomalies being rejected
                if (i % 1000 == 0) and (i != 0):

                    # print anomaly generation result
                    print('{} generated local anomalies were rejected -> consider another seed or increase top frequent values parameter'.format(i))

        # convert to pandas data frame
        local_anomalies = pd.DataFrame(local_anomalies)

        # return created local anomalies
        return local_anomalies


