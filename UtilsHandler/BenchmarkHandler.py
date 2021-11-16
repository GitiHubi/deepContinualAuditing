import numpy as np
from random import shuffle
from avalanche.benchmarks.generators import ni_benchmark


class BenchmarkHandler(object):

    def __init__(self):
        pass

    def get_benchmark(self, params, payment_ds, exp_assignments):
        benchmark = ni_benchmark(payment_ds,
                                 payment_ds,
                                 n_experiences=params["n_experiences"],
                                 fixed_exp_assignment=exp_assignments)

        return benchmark

    def create_percnt_matrix(self, params):
        """ Creates a percentage matrix for a list of classes
            with respect to their change type over time
            ( a: ascending, d: descending, f: fixed)
        """
        dept_percnt, n_experiences = params["dept_percnt"], params["n_experiences"]

        if params["gradual_data_change"]:
            dept_percnt_list = []
            n_depts = len(params["dept_ids"])
            for i in range(n_depts):
                if dept_percnt[i] == 'z':
                    x = np.array(params["percent_type_z"])
                elif dept_percnt[i] == 'a':
                    x = np.array(params["percent_type_a"])
                elif dept_percnt[i] == 'd':
                    x = np.array(params["percent_type_d"])
                elif dept_percnt[i] == 'f':
                    x = np.array(params["percent_type_f"])
                else:
                    raise NotImplementedError()
                dept_percnt_list.append(list(x))

            # transpose the matrix
            dept_percnt_list = np.array(dept_percnt_list).T
            dept_percnt_list = list(dept_percnt_list)
            dept_percnt_list = [list(x) for x in dept_percnt_list]
        else:
            dept_percnt_list = []
            n_depts = len(params["dept_ids"])
            for i in range(n_depts):
                if dept_percnt[i] == 'z':
                    x = np.zeros(n_experiences)
                elif dept_percnt[i] == 'd':
                    x = np.zeros(n_experiences)
                    x[0] = 0.999
                    x[-1] = 0.001
                elif dept_percnt[i] == 'f':
                    x = np.zeros(n_experiences)
                    x[i] = 1.0
                else:
                    raise NotImplementedError()
                dept_percnt_list.append(list(x))

            # transpose the matrix
            dept_percnt_list = np.array(dept_percnt_list).T
            dept_percnt_list = list(dept_percnt_list)
            dept_percnt_list = [list(x) for x in dept_percnt_list]

        # manually set
        dept_percnt_list[-1][-1] = 1.0
        dept_percnt_list[-1][-2] = 1.0

        return dept_percnt_list

    def get_exp_assignment(self, params, payment_ds):
        """ Creates index assignment for each experiment according to the percentage of
        data specified for each dept. id
        """
        if params["load_pecnt_from_params"]:
            data_prcnt = params["dept_percnt"]
        else:
            data_prcnt = self.create_percnt_matrix(params)
        n_experiences = len(data_prcnt)
        ds_dep_indices = {d: list(np.where(payment_ds.payment_depts == d)[0]) for d in params["dept_ids"]}
        ds_dep_count = {d: len(ds_dep_indices[d]) for d in ds_dep_indices.keys()}

        # shuffle ds dept. indices
        _ = [shuffle(ds_dep_indices[d]) for d in ds_dep_indices.keys()]

        # for each experience set how much data to use from each department
        exp_assignments = []  # list of lists that determines which sample ids to use in each experience
        for exp_id in range(n_experiences):
            percentages = data_prcnt[exp_id]
            exp_samples = []
            # for each department compute the number of samples that should be used
            # and assign random samples to the current experience
            for i in range(len(percentages)):
                dept_i = params["dept_ids"][i]
                dept_perc_dep_i = percentages[i]
                n_samples_dept_i = int(dept_perc_dep_i * ds_dep_count[dept_i])
                sampels_i = ds_dep_indices[dept_i][:n_samples_dept_i]
                ds_dep_indices[dept_i] = ds_dep_indices[dept_i][n_samples_dept_i:]
                exp_samples.extend(sampels_i)
            exp_assignments.append(exp_samples)

        return exp_assignments
