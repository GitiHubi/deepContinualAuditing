# import class libraries
import os as os
import json as js
import numpy as np
import pandas as pd
import argparse
import torch

from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (roc_auc_score, average_precision_score)


# class Utilities
class UtilsHandler(object):

    def __init__(self):
        pass

    def create_new_sub_directory(self, timestamp, parent_dir, indicator):

        # create sub directory name
        sub_dir = os.path.join(parent_dir, str(timestamp) + '_' + str(indicator) + '_experiment')

        # case experiment sub directory does not exist
        if not os.path.exists(sub_dir):
            # create new sub directory
            os.makedirs(sub_dir)

        # return new sub directory
        return sub_dir

    def create_experiment_sub_directory(self, parent_dir, folder_name):

        # create sub directory name
        sub_dir = os.path.join(parent_dir, folder_name)

        # case experiment sub directory does not exist
        if (not os.path.exists(sub_dir)):
            # create new sub directory
            os.makedirs(sub_dir)

        # return new sub directory
        return sub_dir

    def create_experiment_directory(self, param, parent_dir, architecture='attention'):

        # case: baseline autoencoder experiment
        if architecture == 'baseline':

            # create experiment directory name
            experiment_directory_name = '{}_NadimBaseline_{}_exp_sd_{}_ep_{}_mb_{}_lr_{}_bt_{}_ds_{}_dv_{}'.format(
                str(param['exp_timestamp']), str(param['architecture']), str(param['seed']), str(param['no_epochs']),
                str(param['batch_size']), str(param['learning_rate']), str(param['bottleneck']), str(param['dataset']), str(param['device']))

        # case: unknown architecture selected
        else:

            # raise exception
            raise Exception('Model architecture is not defined or unknown.')

        # create experiment directory name
        exp_dir = os.path.join(parent_dir, experiment_directory_name)

        # case experiment directory does not exist
        if (not os.path.exists(exp_dir)):
            # create new experiment directory
            os.makedirs(exp_dir)

        # create meta data, signal data, and backtest data sub directories
        par_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='00_param')
        sta_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='01_statistics')
        tra_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='02_training')
        vis_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='03_visuals')
        mod_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='04_models')
        eva_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='05_evaluation')

        # return new experiment directory and sub directories
        return exp_dir, par_sub_dir, sta_sub_dir, tra_sub_dir, vis_sub_dir, mod_sub_dir, eva_sub_dir

    def str2bool(self, value):

        # case: already boolean
        if type(value) == bool:

            # return actual boolean
            return value

        # convert value to lower cased string
        # case: true acronyms
        elif value.lower() in ('yes', 'true', 't', 'y', '1'):

            # return true boolean
            return True

        # case: false acronyms
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):

            # return false boolean
            return False

        # case: no valid acronym detected
        else:

            # rais error
            raise argparse.ArgumentTypeError('[ERROR] Boolean value expected.')

    # save experiment parameter
    def save_experiment_parameter(self, param, parameter_dir):

        # create filename
        filename = str('{}_nadimBaseline_exp_parameter.txt'.format(str(param['exp_timestamp'])))

        # write experimental config to file
        with open(os.path.join(parameter_dir, filename), 'w') as outfile:

            # dump experiment parameters
            js.dump(param, outfile)

    # load experiment parameter
    def load_experiment_parameter(self, parameter_dir):

        # open json parameter file
        parameter_file = open(parameter_dir)

        # load json parameter file
        parameters = js.load(parameter_file)

        # return json parameter
        return parameters

    # save model checkpoint
    def save_checkpoint(self, filename, model, optimizer, iteration, type, chpt_dir):

        # case: baseline autoencoder architecture
        if type == 'baseline':

            # create checkpoint
            checkpoint = {'iter': iteration,
                          'encoder': model.encoder.state_dict(),
                          'decoder': model.decoder.state_dict(),
                          'optim': optimizer.state_dict()
                          }

        # create checkpoint file path
        filepath = os.path.join(chpt_dir, str(filename))

        # save model checkpoint
        torch.save(checkpoint, filepath)

    # collect the number of network parameters
    def get_network_parameter(self, net):

        # init number of parameters
        num_params = 0

        # iterate over net parameters
        for param in net.parameters():

            # collect number of parameters
            num_params += param.numel()

        # return number of network parameters
        return num_params

    # collect the gradient statistics
    def collect_grad_statistics(self, named_parameters):

        # init the to be collected statistics
        ave_grads = []
        max_grads = []
        layers = []

        # iterate over named paramaters
        for name, parameters in named_parameters:

            # collect gradient parameters only
            if (parameters.requires_grad) and ("bias" not in name):

                # determine of parameter contains gradient
                if parameters.grad is not None:

                    # collect layer information
                    layers.append(name)

                    # collect parameter absolute mean of gradients
                    ave_grads.append(parameters.grad.abs().mean().item())

                    # collect parameter absolute max of gradients
                    max_grads.append(parameters.grad.abs().max().item())

        # return gradient statistics
        return layers, ave_grads, max_grads

    def top_K_precision(self, y_score, y_true, k):
        df = pd.concat([y_score, y_true], axis=1)
        df.columns = ['y_score', 'y_true']
        df.sort_values('y_score', ascending=False, inplace=True)
        top_k_labels = df.iloc[:k]['y_true']
        top_k = np.sum(top_k_labels) / k
        return top_k

    def compute_roc_auc_score(self, valid_losses, valid_transactions):

        # compute overall roc-auc score
        roc_auc_score_all = roc_auc_score(y_true=valid_transactions['CLASS'], y_score=valid_losses)

        # compute global roc-auc score
        valid_transactions_global = valid_transactions.copy(deep=True)
        valid_transactions_global.loc[valid_transactions_global['TYPE'] == 'local', 'CLASS'] = 0
        roc_auc_score_global = roc_auc_score(y_true=valid_transactions_global['CLASS'], y_score=valid_losses)

        # compute local roc-auc score
        valid_transactions_local = valid_transactions.copy(deep=True)
        valid_transactions_local.loc[valid_transactions_local['TYPE'] == 'global', 'CLASS'] = 0
        roc_auc_score_local = roc_auc_score(y_true=valid_transactions_local['CLASS'], y_score=valid_losses)

        # return overall, local and global roc-auc scores
        return roc_auc_score_all, roc_auc_score_global, roc_auc_score_local

    def compute_pr_auc_score(self, valid_losses, valid_transactions):

        # compute overall average precision score
        average_precision_score_all = average_precision_score(y_true=valid_transactions['CLASS'], y_score=valid_losses)

        # compute global average precision score
        valid_transactions_global = valid_transactions.copy(deep=True)
        valid_transactions_global.loc[valid_transactions_global['TYPE'] == 'local', 'CLASS'] = 0
        average_precision_score_global = average_precision_score(y_true=valid_transactions_global['CLASS'], y_score=valid_losses)

        # compute local average precision score
        valid_transactions_local = valid_transactions.copy(deep=True)
        valid_transactions_local.loc[valid_transactions_local['TYPE'] == 'global', 'CLASS'] = 0
        average_precision_score_local = average_precision_score(y_true=valid_transactions_local['CLASS'], y_score=valid_losses)

        # return overall, local and global roc-auc scores
        return average_precision_score_all, average_precision_score_global, average_precision_score_local
