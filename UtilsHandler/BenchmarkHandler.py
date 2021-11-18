import numpy as np
import random
import copy
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
            with respect to their reduction type.
        """
        def generate_random_number():
            min = int(params["min_sampling_perc"]*100)
            return random.randint(min, 100) / 100.0

        # Create percentage matrix
        perc_matrix = []
        for dept in params["dept_ids"]:
            if dept in params["target_dept_ids"]:
                perc_matrix.append(params["target_dept_data_perc"])
            elif dept in params["anomaly_dept_ids"]:
                perc_matrix.append(params["anomaly_dept_data_perc"])
            else:

                perc_matrix.append([generate_random_number() for _ in range(params["n_experiences"])])

        # Transpose it
        perc_matrix = np.array(perc_matrix)
        perc_matrix = perc_matrix.T
        perc_matrix = list(perc_matrix)
        perc_matrix = [list(x) for x in perc_matrix]

        return perc_matrix

    def get_exp_assignment(self, params, payment_ds, perc_matrix):
        """ Creates index assignment for each experiment according to the percentage of
        data specified for each dept. id
        """
        n_experiences = params["n_experiences"]

        # create list of sample indices per deptartment
        ds_dep_indices = {d: list(np.where(payment_ds.payment_depts == d)[0]) for d in params["dept_ids"]}

        # shuffle ds dept. indices
        _ = [random.shuffle(ds_dep_indices[d]) for d in ds_dep_indices.keys()]

        # for each experience set how much data to use from each department
        exp_assignments = []  # list of lists that determines which sample ids to use in each experience
        for exp_id in range(n_experiences):
            percentages = perc_matrix[exp_id]
            experience_i_indices = []  # list of experience indices

            # for each department compute the number of samples that should be used
            # and assign random samples to the current experience
            for i, dept_id in enumerate(params["dept_ids"]):
                dept_i_perc = percentages[i]
                # if anomaly:
                if dept_id in params["anomaly_dept_ids"]:
                    # for anomaly departments only add in the final experience
                    if exp_id < n_experiences - 1:
                        continue
                    dept_i_indices = copy.copy(ds_dep_indices[dept_id][:])
                    subset_max_idx = int(dept_i_perc * len(dept_i_indices))
                    experience_i_indices.extend(dept_i_indices[:subset_max_idx])

                # otherwise:
                else:
                    start = exp_id * params["n_dept_samples_per_exp"]
                    end = (exp_id+1) * params["n_dept_samples_per_exp"]
                    dept_i_indices = copy.copy(ds_dep_indices[dept_id][start:end])
                    subset_max_idx = int(dept_i_perc * len(dept_i_indices))
                    experience_i_indices.extend(dept_i_indices[:subset_max_idx])

            exp_assignments.append(experience_i_indices)

        return exp_assignments
