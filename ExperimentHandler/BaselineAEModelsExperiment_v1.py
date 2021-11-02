from __future__ import print_function

# import os library
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import pytorch libraries
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# import multiprocessing libraries
from multiprocessing import Process

# import sklearn libraries
from sklearn import metrics

# import additional libraries
import warnings
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ignore all warnings
warnings.filterwarnings("ignore")

# import network architectures
import NetworkHandler.BaselineAutoencoder as BaselineAutoencoder

# import and init utilities handler
import UtilsHandler.UtilsHandler as UtilsHandler
uha = UtilsHandler.UtilsHandler()

# import and init data handler
import DataHandler.DataHandler as DataHandler
dha = DataHandler.DataHandler()

# import and init payment dataset
import DataHandler.PaymentDataset as PaymentDataset

# init visualization handler
#import VisualizationHandler.VisualizationHandler as VisualizationHandler
#vha = VisualizationHandler.VisualizationHandler()

# define baseline autoencoder experiment
def run_baseline_autoencoder_experiment(experiment_parameter):

    # log and print training run configuration
    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
    print('[INFO {}] Deep Nadim Baseline Experiment :: experiment parameter: {}.'.format(now, str(experiment_parameter)))
    print('[INFO {}] Deep Nadim Baseline Experiment :: network training device : {}.'.format(now, str(experiment_parameter['device'])))

    ########################################################################################################################
    # A. prepare experiment directories
    ########################################################################################################################

    # create the experiment directory
    experiment_parameter['experiment_dir'], experiment_parameter['par_dir'], experiment_parameter['sta_dir'], experiment_parameter['tra_dir'], experiment_parameter['plt_dir'], experiment_parameter['mod_dir'], experiment_parameter['eva_dir'] = uha.create_experiment_directory(param=experiment_parameter, parent_dir=experiment_parameter['base_dir'], architecture=experiment_parameter['architecture'])

    # log and save experiment parameters
    uha.save_experiment_parameter(param=experiment_parameter, parameter_dir=experiment_parameter['par_dir'])

    ########################################################################################################################
    # C. prepare experiment statistics
    ########################################################################################################################

    # init baseline training experiment statistics
    summary_cols = [
        'timestamp'
        , 'seed'
        , 'task'
        , 'no_epochs'
        , 'batch_size'
        , 'learning_start'
        , 'learning_rate'
        , 'weight_decay'
        , 'epoch'
        , 'train_rec_loss'
        , 'eval_rec_loss_dept'
    ]
    experiment_training_statistics = pd.DataFrame(columns=summary_cols)

    ########################################################################################################################
    # E. prepare and initialize model training
    ########################################################################################################################

    # case: categorical mse loss
    if experiment_parameter['categorical_loss'] == 'bce':

        # init aggregated categorical autoencoder loss
        rec_criterion_cat = torch.nn.BCELoss().to(experiment_parameter['device'])

        # init detailed categorical autoencoder loss
        rec_criterion_cat_details = torch.nn.MSELoss(reduction='none').to(experiment_parameter['device'])

    # init last model checkpoint name
    model_checkpoint_name = -1

    # iterate over number of training tasks
    for task in range(1, experiment_parameter['no_tasks'] + 1):

        # load and prepare philadelphia city payment data
        transactions_dataset, transactions_encoded, transactions_encoded_ids = dha.get_Philadelphia_data(experiment_parameters=experiment_parameter, task_id=task)

        # init transactions training dataset
        transactions_encoded_training = PaymentDataset.PaymentDataset(payments_encoded=transactions_encoded, payment_ids=transactions_encoded_ids, task_id=task)

        # init training data loader
        data_loader = DataLoader(transactions_encoded_training, batch_size=experiment_parameter['batch_size'], num_workers=experiment_parameter['no_workers'], drop_last=False, shuffle=False)

        # case: initial training task
        if task == 1:

            # init autoencoder baseline model
            model = init_baseline_autoencoder_model(experiment_parameter, transactions_encoded)

        # case: non-initial training task
        else:

            # load last model checkpoint file
            checkpoint_file = torch.load(os.path.join(experiment_parameter['mod_dir'], model_checkpoint_name), map_location=lambda storage, loc: storage)

            # load encoder and decoder parameter
            model.encoder.load_state_dict(checkpoint_file['encoder'])
            model.decoder.load_state_dict(checkpoint_file['decoder'])

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] Deep Nadim Baseline Experiment :: task {} AE baseline checkpoint {} model successfully loaded.'.format(now, str(task), str(model_checkpoint_name)))

        # push baseline autoencoder model to compute device
        model = model.to(experiment_parameter['device'])

        # init optimizer
        optimizer = optim.Adam(params=model.parameters(), lr=experiment_parameter['learning_rate'], weight_decay=experiment_parameter['weight_decay'])

        # init learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=experiment_parameter['no_epochs'], eta_min=0, last_epoch=-1)

        # run model task training
        model_checkpoint_name, experiment_training_statistics = run_baseline_autoencoder_training_task(experiment_parameter, transactions_dataset, task, model, data_loader, optimizer, scheduler, experiment_training_statistics, rec_criterion_cat, rec_criterion_cat_details)

def run_baseline_autoencoder_training_task(experiment_parameter, transactions_dataset, task, model, data_loader, optimizer, scheduler, experiment_training_statistics, rec_criterion_cat, rec_criterion_cat_details):

    # iterate over number of training samples
    for i in range(experiment_parameter['no_epochs']):

        #### run the model training

        # set model to training mode
        model.train()

        # run model epoch training
        epoch_train_loss, model = run_baseline_autoencoder_training_epoch(experiment_parameter=experiment_parameter, epoch=i, model=model, optimizer=optimizer, loader=data_loader, rec_criterion_cat=rec_criterion_cat)

        # case: model eval epoch
        if i % experiment_parameter['eval_epoch'] == 0:

            #### run the model evaluation

            # set model to evaluation mode
            model.eval()

            # run model evaluation
            epoch_valid_losses = run_baseline_autoencoder_evaluation_epoch(experiment_parameter=experiment_parameter, epoch=i, model=model, loader=data_loader, rec_criterion_cat=rec_criterion_cat_details)

            # merge transaction dataset and corresponding losses
            transactions_dataset_rec_losses = transactions_dataset.join(epoch_valid_losses, lsuffix='_transaction', rsuffix='_losses')

            # determine mean reconstruction loss per department
            transactions_dataset_mean_department_rec_losses = transactions_dataset_rec_losses.groupby(['dept']).mean()['rec_error']

        # collect the training statistics of current epoch
        training_stats = {'timestamp': str(dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')),
                          'seed': experiment_parameter['seed'],
                          'task': str(task),
                          'no_epochs': experiment_parameter['no_epochs'],
                          'batch_size':  experiment_parameter['batch_size'],
                          'learning_start': experiment_parameter['learning_rate'],
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          'weight_decay': experiment_parameter['weight_decay'],
                          'epoch': str(i),
                          'train_rec_loss': np.round(epoch_train_loss, 8),
                          'eval_rec_loss_dept': dict(np.round(transactions_dataset_mean_department_rec_losses, 8))
                          }

        # append the training statistics of current epoch
        experiment_training_statistics = experiment_training_statistics.append(training_stats, ignore_index=True)

        # save current experiment statistics
        file_name = '{}_experiment_training_statistics_sd_{}_ep_{}.csv'.format(experiment_parameter['exp_timestamp'], experiment_parameter['seed'], experiment_parameter['no_epochs'])
        experiment_training_statistics.to_csv(os.path.join(experiment_parameter['sta_dir'], file_name), sep=',', encoding='utf-8')

        # case: model checkpoint epoch
        if (i % experiment_parameter['checkpoint_epoch'] == 0) and (experiment_parameter['checkpoint_save'] == True):

            # save model checkpoint
            model_checkpoint_name = 'deepNadim_baseline_checkpoint_ta_{}_ep_{}.pth'.format(str(task), str(i).zfill(6))
            uha.save_checkpoint(filename=model_checkpoint_name, model=model, optimizer=optimizer, iteration=(i), type='baseline', chpt_dir=experiment_parameter['mod_dir'])

        # case: current epoch non warm-up epoch
        if i >= experiment_parameter['warmup_epochs']:

            # update learning rate
            scheduler.step()

    # return last model checkpoint name
    return model_checkpoint_name, experiment_training_statistics

def run_baseline_autoencoder_training_epoch(experiment_parameter, epoch, model, loader, optimizer, rec_criterion_cat):

    # init epoch training loss
    epoch_train_loss = 0.0
    epoch_train_rec_loss = 0.0

    # init iteration training loss
    iteration_train_losses = []

    # wrap training loader into progress bar
    epoch_train_loader = tqdm(loader)

    # init number of batch iterations
    batch_count = 0

    # iterate over epoch training mini-batches
    for _, batch in epoch_train_loader:

        # push both mini-batches to compute device
        batch = batch.to(experiment_parameter['device'], non_blocking=True)

        # get the reconstructions and representations
        _, rec_batch = model(batch)

        # init attribute scaled reconstruction loss
        rec_loss_cat = torch.zeros(1).to(experiment_parameter['device'])
        # rec_loss_num = torch.zeros(1).to(experiment_parameter['device'])

        # determine start, end, and size of encoded categorical attribute column
        # for attribute in encoded_columns_map['cat_attributes']:

            # determine start, end, and size of encoded attribute column
            # col_start = encoded_columns_map[str(attribute) + '_start']
            # col_end = encoded_columns_map[str(attribute) + '_end']
            # col_size = encoded_columns_map[str(attribute) + '_size']

        # determine weighted reconstruction error of current categorical attribute column
        # rec_loss_single_cat_attribute = rec_criterion_cat(input=rec_batch[:, col_start:col_end], target=batch[:, col_start:col_end]) # * col_size
        rec_loss_single_cat_attribute = rec_criterion_cat(input=rec_batch, target=batch) # * col_size

        # collect weighted reconstruction error of current categorical attribute column
        rec_loss_cat += rec_loss_single_cat_attribute

        # determine start, end, and size of encoded numerical attribute column
        #for attribute in encoded_columns_map['num_attributes']:

            # determine start, end, and size of encoded attribute column
            #col_start = encoded_columns_map[str(attribute) + '_start']
            #col_end = encoded_columns_map[str(attribute) + '_end']
            # col_size = encoded_columns_map[str(attribute) + '_size']

            # determine weighted reconstruction error of current numerical attribute column
            #rec_loss_single_num_attribute = rec_criterion_num(input=rec_batch[:, col_start:col_end], target=batch[:, col_start:col_end]) # * col_size

            # collect weighted reconstruction error of current numerical attribute column
            #rec_loss_num += rec_loss_single_num_attribute

        # combine both reconstruction errors
        rec_loss = rec_loss_cat # + rec_loss_num

        # reset model gradients
        optimizer.zero_grad()

        # determine gradients
        rec_loss.backward()

        # optimize network parameters
        optimizer.step()

        # collect training losses
        iteration_train_losses.append(rec_loss.item())

        # collect and update training loss
        epoch_train_loss += rec_loss.item()

        # determine timestamp
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')

        # log training progress
        epoch_train_loader.set_description(
            (
                '[INFO {}] Deep Nadim Baseline Experiment [{}/{}] :: train-loss: {:3f}.'.format(str(now), str(epoch), str(experiment_parameter['no_epochs']), np.mean(iteration_train_losses[-100:]))
            )
        )

        # init number of batch iterations
        batch_count += 1

    # compute overall training loss
    epoch_train_loss = epoch_train_loss / batch_count

    # return training results
    return epoch_train_loss, model

def run_baseline_autoencoder_evaluation_epoch(experiment_parameter, epoch, model, loader, rec_criterion_cat):

    # init iteration reconstructed ids
    rec_loss_cat_ids = []

    # init iteration reconstruction losses
    rec_loss_cat_individual = []

    # wrap training loader into progress bar
    epoch_valid_loader = tqdm(loader)

    # iterate over epoch training mini-batches
    for ids, batch in epoch_valid_loader:

        # determine individual reconstructed ids
        rec_loss_cat_ids = np.concatenate((rec_loss_cat_ids, ids.detach().numpy()))

        # push both mini-batches to compute device
        batch = batch.to(experiment_parameter['device'], non_blocking=True)

        # get the reconstructions and representations
        _, rec_batch = model(batch)

        # determine individual reconstruction losses per transaction
        rec_loss_individual_batch = torch.mean(rec_criterion_cat(input=rec_batch, target=batch), axis=1)

        # collect individual transactional reconstruction losses
        rec_loss_cat_individual = np.concatenate((rec_loss_cat_individual, rec_loss_individual_batch.detach().numpy()))

        # determine timestamp
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')

        # log training progress
        epoch_valid_loader.set_description(
            (
                '[INFO {}] Deep Nadim Baseline Experiment [{}/{}] :: eval-loss: {:3f}.'.format(str(now), str(epoch), str(experiment_parameter['no_epochs']), np.mean(rec_loss_individual_batch.detach().numpy()))
            )
        )

    # merge reconstructed ids and corresponding losses
    rec_loss_individual = pd.DataFrame({'id': rec_loss_cat_ids.astype(int), 'rec_error': rec_loss_cat_individual.astype(float)})

    # return training results
    return rec_loss_individual

def init_baseline_autoencoder_model(experiment_parameter, transactions_encoded):

    # update the encoder and decoder network input dimensionality depending on the training data
    experiment_parameter['encoder_dim'].insert(0, transactions_encoded.shape[1])
    experiment_parameter['decoder_dim'].insert(len(experiment_parameter['decoder_dim']), transactions_encoded.shape[1])

    # init the baseline autoencoder model
    model = BaselineAutoencoder.BaselineAutoencoder(
        encoder_layers=experiment_parameter['encoder_dim'],
        encoder_bottleneck=experiment_parameter['bottleneck'],
        decoder_layers=experiment_parameter['decoder_dim']
    )

    # determine decoder number of parameters
    model_parameter = uha.get_network_parameter(net=model)

    # log configuration processing
    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
    print('[INFO {}] Deep Nadim Baseline Experiment :: AE baseline model: {}.'.format(now, str(model)))
    print('[INFO {}] Deep Nadim Baseline Experiment :: Total AE baseline model parameters: {}.'.format(now, str(model_parameter)))

    # return autoencoder baseline model
    return model

