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
import yaml
import wandb
import time

from avalanche.benchmarks.generators import ni_benchmark
from avalanche.training.strategies import JointTraining
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, WandBLogger

import UtilsHandler.UtilsHandler as UtilsHandler
from DataHandler.payment_dataset_philadelphia import PaymentDatasetPhiladephia
import NetworkHandler.BaselineAutoencoder as BaselineAutoencoder


def load_params(yml_file_path):
    """ Loads param file and returns a dictionary. """
    with open(yml_file_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)

    return params


def get_model(experiment_parameter, transactions_encoded):
    # update the encoder and decoder network input dimensionality depending on the training data
    experiment_parameter['encoder_dim'].insert(0, transactions_encoded.shape[1])
    experiment_parameter['decoder_dim'].insert(len(experiment_parameter['decoder_dim']),
                                               transactions_encoded.shape[1])

    # init the baseline autoencoder model
    model = BaselineAutoencoder.BaselineAutoencoder(
        encoder_layers=experiment_parameter['encoder_dim'],
        encoder_bottleneck=experiment_parameter['bottleneck'],
        decoder_layers=experiment_parameter['decoder_dim']
    )

    # return autoencoder baseline model
    return model


def get_strategy(experiment_parameters, payment_ds):
    # initialize evaluator for metrics and loggers
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger]
    )

    # initialize model
    model = get_model(experiment_parameters, payment_ds.payments_encoded)

    # initialize optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=experiment_parameter['learning_rate'],
                                 weight_decay=experiment_parameter['weight_decay'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experiment_parameters["strategy"] == "JointTraining":
        strategy = JointTraining(model=model,
                         optimizer=optimizer,
                         criterion=torch.nn.BCELoss(),
                         train_mb_size=experiment_parameters["batch_size"],
                         train_epochs=experiment_parameters["no_epochs"],
                         evaluator=eval_plugin,
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
    params = load_params(experiment_parameters["params_path"])
    benchmark = ni_benchmark(payment_ds,
                             payment_ds,
                             n_experiences=params["n_experiments"])

    # get strategy
    strategy = get_strategy(experiment_parameters, payment_ds)

    # initialize WandB
    run_name = experiment_parameters["strategy"] + "_nexp" + str(params["n_experiments"])
    run_name += "_" + time.strftime("%Y%m%d%H%M%S")
    log_wandb = args.wandb_proj != ''
    if log_wandb:
        wandb.init(
            project=args.wandb_proj,
            config=experiment_parameters,
            id=run_name)
        wandb.run.name = run_name

        # create folder for the current experiment
        output_path = os.path.join(args.outputs_path, run_name)
        os.makedirs(output_path, exist_ok=True)

    global_iter = 0

    # joint train on all experiences
    res_train = strategy.train(benchmark.train_stream)
    loss_train_exp = res_train[f"Loss_Epoch/train_phase/train_stream/Task000"]

    if log_wandb:
        wandb.log({"experience/loss_train": loss_train_exp}, step=global_iter)

    # eval loss for all experiences
    res_eval = strategy.eval(benchmark.train_stream)
    for exp_id in range(len(benchmark.train_stream)):
        loss_eval_exp = res_eval[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{exp_id:03d}"]
        if log_wandb:
            wandb.log({f"experience/loss_exp_{exp_id}": loss_eval_exp}, step=global_iter)

    # compute average loss on all experiences and save the checkpoint
    loss_eval_exp_allseen = res_eval[f"Loss_Stream/eval_phase/train_stream/Task000"]
    if log_wandb:
        wandb.log({"experience/loss_exp_allseen": loss_eval_exp_allseen}, step=global_iter)
        torch.save(strategy.model.state_dict(), os.path.join(output_path, f"ckpt_0.pt"))
        wandb.finish()


# define main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deepNadim Experiments')

    # experiment parameter
    parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)
    parser.add_argument('--base_dir', help='', nargs='?', type=str, default='./200_experiments/01_baselines/001_experiment_track')

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
    parser.add_argument('--wandb_proj', help='', nargs='?', type=str, default='')
    parser.add_argument('--n_exp', help='', nargs='?', type=int, default=10)

    parser.add_argument('--params_path', help='', nargs='?', type=str, default='params/params.yml')
    parser.add_argument('--outputs_path', help='', nargs='?', type=str, default='./outputs')


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
