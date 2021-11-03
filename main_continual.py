import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# imports
import argparse
import numpy as np
import torch

from avalanche.benchmarks.generators import ni_benchmark
from avalanche.training.strategies import Naive, EWC
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger, WandBLogger

import UtilsHandler.UtilsHandler as UtilsHandler
from DataHandler.payment_dataset_philadelphia import PaymentDatasetPhiladephia
import NetworkHandler.BaselineAutoencoder as BaselineAutoencoder


def get_model(experiment_parameter, transactions_encoded):
    # update the encoder and decoder network input dimensionality depending on the training data
    experiment_parameter['encoder_dim'].insert(0, transactions_encoded.shape[1])
    experiment_parameter['decoder_dim'].insert(len(experiment_parameter['decoder_dim']), transactions_encoded.shape[1])

    # init the baseline autoencoder model
    model = BaselineAutoencoder.BaselineAutoencoder(
        encoder_layers=experiment_parameter['encoder_dim'],
        encoder_bottleneck=experiment_parameter['bottleneck'],
        decoder_layers=experiment_parameter['decoder_dim']
    )

    # return autoencoder baseline model
    return model


def ae_criterion(out, target):
    """ Criterion function for the autoencoder model. It splits the out to z, pred
        correspondingly and then computes the binary cross-entropy loss.
    """
    _, pred = out
    return torch.nn.BCELoss()(pred, target)


def get_strategy(experiment_parameters, payment_ds, args):
    # initialize evaluator for metrics and loggers
    interactive_logger = InteractiveLogger()

    run_name = experiment_parameters["strategy"] + "_nexp" + str(experiment_parameters["n_exp"])
    wandb_logger = WandBLogger(project_name="ContinualAnomaly", run_name=run_name, config=args)

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, wandb_logger]
    )

    # initialize model
    model = get_model(experiment_parameters, payment_ds.payments_encoded)

    # initialize optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=experiment_parameter['learning_rate'],
                                 weight_decay=experiment_parameter['weight_decay'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experiment_parameters["strategy"] == "Naive":
        strategy = Naive(model=model,
                         optimizer=optimizer,
                         criterion=ae_criterion,
                         train_mb_size=experiment_parameters["batch_size"],
                         train_epochs=experiment_parameters["no_epochs"],
                         evaluator=eval_plugin,
                         device=device)

    elif experiment_parameters["strategy"] == "EWC":
        strategy = EWC(model=model,
                       optimizer=optimizer,
                       criterion=ae_criterion,
                       ewc_lambda=500,
                       train_mb_size=experiment_parameters["batch_size"],
                       train_epochs=experiment_parameters["no_epochs"],
                       evaluator=eval_plugin,
                       device=device)

    elif experiment_parameters["strategy"] == "Replay":
        replay_plugin = ReplayPlugin(mem_size=500)
        strategy = Naive(model=model,
                       optimizer=optimizer,
                       criterion=ae_criterion,
                       train_mb_size=experiment_parameters["batch_size"],
                       train_epochs=experiment_parameters["no_epochs"],
                       evaluator=eval_plugin,
                       plugins=[replay_plugin],
                       device=device)

    else:
        raise NotImplementedError()

    return strategy


def main(experiment_parameters, args):
    """ Main function: initializes the dataset and creates benchmark for
        continual learning. Then, it loops over all experiences and trains/evaluates
        the model using a defined strategy.
    """
    # initialize payment dataset
    payment_ds = PaymentDatasetPhiladephia(experiment_parameters['data_dir'])

    # create an instance-incremental benchmark by which new samples become available over time
    benchmark = ni_benchmark(payment_ds, payment_ds, n_experiences=experiment_parameters["n_exp"])

    # get strategy
    strategy = get_strategy(experiment_parameters, payment_ds, args)

    # iterate through all experiences (tasks) and train the model over each experience
    for exp_id, exp in enumerate(benchmark.train_stream):
        strategy.train(exp)
        strategy.eval(benchmark.train_stream[:exp_id+1])


# define main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='deepNadim Experiments')

    # experiment parameter
    parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)
    parser.add_argument('--base_dir', help='', nargs='?', type=str, default='././200_experiments/01_baselines/001_experiment_track')

    # dataset and data loading parameter
    parser.add_argument('--dataset', help='', nargs='?', type=str, default='philadelphia') # chicago, philadelphia
    parser.add_argument('--data_dir', help='', nargs='?', type=str, default='./100_datasets/philadelphia') # chicago, philadelphia
    parser.add_argument('--no_workers', help='', nargs='?', type=int, default=0)

    # encoder architecture parameter
    parser.add_argument('--architecture', help='', nargs='?', type=str, default='baseline')
    parser.add_argument('--bottleneck', help='', nargs='?', type=str, default='linear')  ## lrelu, tanh, linear

    # architecture parameter
    #parser.add_argument('--encoder_dim', help='', nargs='+', default=[4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2])
    #parser.add_argument('--decoder_dim', help='', nargs='+', default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    parser.add_argument('--encoder_dim', help='', nargs='+', default=[128, 64, 32, 16, 8, 4, 2])
    parser.add_argument('--decoder_dim', help='', nargs='+', default=[2, 4, 8, 16, 32, 64, 128])

    # training parameter
    parser.add_argument('--batch_size', help='', nargs='?', type=int, default=128)
    parser.add_argument('--no_epochs', help='', nargs='?', type=int, default=10)
    parser.add_argument('--no_tasks', help='', nargs='?', type=int, default=6)
    parser.add_argument('--warmup_epochs', help='', nargs='?', type=float, default=100)
    parser.add_argument('--optimizer', help='', nargs='?', type=str, default='adam')
    parser.add_argument('--learning_rate', help='', nargs='?', type=float, default=1e-4)
    parser.add_argument('--eval_epoch', help='', nargs='?', type=int, default=1)
    parser.add_argument('--weight_decay', help='', nargs='?', type=float, default=1e-6)

    # loss parameter
    parser.add_argument('--categorical_loss', help='', nargs='?', type=str, default='bce')  # mse, bce

    # evaluation parameter
    parser.add_argument('--valid_size', help='', nargs='?', type=float, default=1.00)  # 238894
    parser.add_argument('--sample_test', help='', type=str, default='True')
    parser.add_argument('--sample_size', help='', nargs='?', type=int, default=10)

    # number of created artificial anomalies
    parser.add_argument('--global_anomalies', help='', nargs='?', type=int, default=60)  # 60
    parser.add_argument('--local_anomalies', help='', nargs='?', type=int, default=140)  # 140

    # logging parameter
    parser.add_argument('--wandb_logging', help='', type=str, default='True')
    parser.add_argument('--checkpoint_epoch', help='', nargs='?', type=int, default=1)
    parser.add_argument('--checkpoint_save', help='', type=str, default='True')

    # new
    parser.add_argument('--strategy', help='', nargs='?', type=str, default='Naive')
    parser.add_argument('--wandb_proj', help='', nargs='?', type=str, default='ContinualAnomaly')
    parser.add_argument('--n_exp', help='', nargs='?', type=int, default=10)

    # parse script arguments
    args = parser.parse_args()
    experiment_parameter = vars(parser.parse_args())

    # determine hardware device
    experiment_parameter['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type

    # set deterministic seeds of the experiments
    seed_value = experiment_parameter['seed']
    np.random.seed(seed_value)

    # case: cuda enabled
    if experiment_parameter['device'] == 'cuda:0':

        # set deterministic cuda backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # set seeds cuda backend
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)

    # init utilities handler
    uha = UtilsHandler.UtilsHandler()

    # parse boolean args as boolean
    experiment_parameter['checkpoint_save'] = uha.str2bool(experiment_parameter['checkpoint_save'])
    experiment_parameter['wandb_logging'] = uha.str2bool(experiment_parameter['wandb_logging'])

    # parse string args as int
    experiment_parameter['encoder_dim'] = [int(ele) for ele in experiment_parameter['encoder_dim']]
    experiment_parameter['decoder_dim'] = [int(ele) for ele in experiment_parameter['decoder_dim']]

    # case: baseline autoencoder experiment
    if experiment_parameter['architecture'] == 'baseline':
        main(experiment_parameter, args)
