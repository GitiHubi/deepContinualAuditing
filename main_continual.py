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
from scipy.special import softmax
import numpy as np
import torch
import yaml
from random import shuffle
import wandb
import time

from avalanche.benchmarks.generators import ni_benchmark
from avalanche.training.strategies import Naive, EWC, LwF, SynapticIntelligence
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
import copy

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

def ae_criterion(out, target):
    """ Criterion function for the autoencoder model. It splits the out to z, pred
        correspondingly and then computes the binary cross-entropy loss.
    """
    _, pred = out
    return torch.nn.BCELoss()(pred, target)


def main(experiment_parameters):
    """ Main function: initializes the dataset and creates benchmark for
        continual learning. Then, it loops over all experiences and trains/evaluates
        the model using a defined strategy.
    """
    # initialize payment dataset
    payment_ds = PaymentDatasetPhiladephia(experiment_parameters['data_dir'])

    # create an instance-incremental benchmark by which new samples become available over time
    benchmark = ni_benchmark(payment_ds, payment_ds, n_experiences=5)

    # initialize model
    model = get_model(experiment_parameters, payment_ds.payments_encoded)

    # initialize optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=experiment_parameters['learning_rate'],
                                 weight_decay=experiment_parameters['weight_decay'])

def create_percnt_matrix(dept_ids, params):
    """ Creates a percentage matrix for a list of classes
        with respect to their change type over time
        ( a: ascending, d: descending, f: fixed)
    """
    dept_percnt, n_experiments =  params["dept_percnt"], params["n_experiments"]

    dept_percnt_list = []
    n_depts = len(dept_ids)
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

    # manually set
    dept_percnt_list[-1][-1] = 1.0
    dept_percnt_list[-1][-2] = 1.0

    return dept_percnt_list


def get_exp_assignment(params, payment_ds):
    """ Creates index assignment for each experiment according to the percentage of
    data specified for each dept. id
    """
    dept_ids = params["dept_ids"]
    create_dept_percnt_matrix = params["create_dept_percnt_matrix"]

    if create_dept_percnt_matrix:
        data_prcnt = create_percnt_matrix(dept_ids, params)
    else:
        data_prcnt = params["dept_percnt"]

    n_experiences = len(data_prcnt)

    ds_dep_indices = {d: list(np.where(payment_ds.payment_depts == d)[0]) for d in dept_ids}
    ds_dep_count = {d: len(ds_dep_indices[d]) for d in ds_dep_indices.keys()}

    # shuffle ds dept. indices
    _ = [shuffle(ds_dep_indices[d]) for d in ds_dep_indices.keys()]

    exp_assignments = []
    for exp_id in range(n_experiences):
        percentages = data_prcnt[exp_id]
        exp_samples = []
        for i in range(len(percentages)):
            dept_i = dept_ids[i]
            dept_perc_dep_i = percentages[i]
            n_samples_dept_i = int(dept_perc_dep_i * ds_dep_count[dept_i])
            sampels_i = ds_dep_indices[dept_i][:n_samples_dept_i]
            ds_dep_indices[dept_i] = ds_dep_indices[dept_i][n_samples_dept_i:]
            exp_samples.extend(sampels_i)
        exp_assignments.append(exp_samples)

    return exp_assignments


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
    if experiment_parameters["strategy"] == "Naive":
        strategy = Naive(model=model,
                         optimizer=optimizer,
                         criterion=torch.nn.BCELoss(),
                         train_mb_size=experiment_parameters["batch_size"],
                         train_epochs=experiment_parameters["no_epochs"],
                         evaluator=eval_plugin,
                         device=device)

    elif experiment_parameters["strategy"] == "EWC":
        strategy = EWC(model=model,
                       optimizer=optimizer,
                       criterion=torch.nn.BCELoss(),
                       ewc_lambda=experiment_parameters["ewc_lambda"],
                       train_mb_size=experiment_parameters["batch_size"],
                       train_epochs=experiment_parameters["no_epochs"],
                       evaluator=eval_plugin,
                       device=device)

    elif experiment_parameters["strategy"] == "LwF":
        strategy = LwF(model=model,
                       optimizer=optimizer,
                       criterion=torch.nn.BCELoss(),
                       alpha=experiment_parameters["lwf_alpha"],
                       temperature=experiment_parameters["lwf_temperature"],
                       train_mb_size=experiment_parameters["batch_size"],
                       train_epochs=experiment_parameters["no_epochs"],
                       evaluator=eval_plugin,
                       device=device)

    elif experiment_parameters["strategy"] == "SynapticIntelligence":
        strategy = SynapticIntelligence(model=model,
                       optimizer=optimizer,
                       criterion=torch.nn.BCELoss(),
                       si_lambda=experiment_parameters["si_lambda"],
                       eps=experiment_parameters["si_eps"],
                       train_mb_size=experiment_parameters["batch_size"],
                       train_epochs=experiment_parameters["no_epochs"],
                       evaluator=eval_plugin,
                       device=device)

    elif experiment_parameters["strategy"] == "Replay":
        replay_plugin = ReplayPlugin(mem_size=experiment_parameters["replay_mem_size"])
        strategy = Naive(model=model,
                         optimizer=optimizer,
                         criterion=torch.nn.BCELoss(),
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
    params = load_params(experiment_parameters["params_path"])
    exp_assignments = get_exp_assignment(params, payment_ds)
    benchmark = ni_benchmark(payment_ds,
                             payment_ds,
                             n_experiences=params["n_experiments"],
                             fixed_exp_assignment=exp_assignments)

    # get strategy
    strategy = get_strategy(experiment_parameters, payment_ds)

    # initialize WandB
    run_name = params["scenario"] + "_nexp" + str(params["n_experiments"]) + "_" + experiment_parameters["strategy"]
    if params["train_only_on_last_experience"]:
        run_name += "_ONLYFINALEXP"

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
    # iterate through all experiences (tasks) and train the model over each experience
    for exp_id, exp in enumerate(benchmark.train_stream):
        # skip for N-1 experiences
        if params["train_only_on_last_experience"] == True and exp_id < len(benchmark.train_stream)-1:
            continue
        res_train = strategy.train(exp)
        loss_train_exp = res_train[f"Loss_Epoch/train_phase/train_stream/Task000"]

        # eval loss for the current experiment
        res_eval = strategy.eval(benchmark.train_stream[:exp_id+1])
        loss_eval_exp = res_eval[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{exp_id:03d}"]
        loss_eval_exp_allseen = res_eval[f"Loss_Stream/eval_phase/train_stream/Task000"]

        if log_wandb:
            wandb.log({"experience/loss_train": loss_train_exp}, step=global_iter)
            wandb.log({"experience/loss_exp": loss_eval_exp}, step=global_iter)
            wandb.log({"experience/loss_exp_allseen": loss_eval_exp_allseen}, step=global_iter)

        # compute per-department loss for all seen experiences
        loss_per_dep = [[] for i in params["dept_ids"]]
        for i in range(exp_id+1):
            exp_i = benchmark.train_stream[i]
            main_test_ds_i = copy.copy(exp_i.dataset)
            for itr_dep, dept_id in enumerate(params["dept_ids"]):
                dept_indices = torch.where(main_test_ds_i[:][3] == dept_id)
                subexp_ds = AvalancheSubset(main_test_ds_i, indices=dept_indices[0])
                if len(subexp_ds) > 0:
                    exp_i.dataset = subexp_ds
                    res = strategy.eval(exp_i)
                    loss_dept_i = res[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{i:03d}"]
                else:
                    loss_dept_i = None
                loss_per_dep[itr_dep].append(loss_dept_i)

        if log_wandb:
            for itr_dep, dept_id in enumerate(params["dept_ids"]):
                dep_losses = loss_per_dep[itr_dep]
                if dep_losses[-1] != None:
                    wandb.log({f"dept/loss_dept{dept_id}": dep_losses[-1]}, step=global_iter)
                dep_losses = [l for l in dep_losses if l != None]
                if len(dep_losses) > 0:
                    wandb.log({f"dept/loss_dept{dept_id}_avg": np.mean(dep_losses)}, step=global_iter)

            torch.save(strategy.model.state_dict(), os.path.join(output_path, f"ckpt_{exp_id}.pt"))

        global_iter += 1

    if log_wandb:
        wandb.finish()

# define main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='deepNadim Experiments')

    # experiment parameter
    parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)


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

    # parse script arguments
    parser.add_argument('--strategy', help='', nargs='?', type=str, default='Naive')
    parser.add_argument('--wandb_proj', help='', nargs='?', type=str, default='')

    parser.add_argument('--params_path', help='', nargs='?', type=str, default='params/params.yml')
    parser.add_argument('--outputs_path', help='', nargs='?', type=str, default='./outputs')

    # ==========
    # ========== Strategies
    # ==========
    # Replay
    parser.add_argument('--replay_mem_size', help='', nargs='?', type=int, default=500)  # 238894

    # lwf
    parser.add_argument('--lwf_alpha', help='', nargs='?', type=float, default=1.00)  # 238894
    parser.add_argument('--lwf_temperature', help='', nargs='?', type=float, default=1.00)  # 238894

    # ewc
    parser.add_argument('--ewc_lambda', help='', nargs='?', type=float, default=1.00)

    # synaptic intelligence
    parser.add_argument('--si_lambda', help='', nargs='?', type=float, default=1.00)
    parser.add_argument('--si_eps', help='', nargs='?', type=float, default=0.001)


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
