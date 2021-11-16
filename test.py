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
import torch
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import seaborn as sns

from avalanche.benchmarks.generators import ni_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
import copy

import UtilsHandler.UtilsHandler as UtilsHandler
from DataHandler.PaymentDataset import PaymentDatasetPhiladephia

from main_continual import load_params, get_exp_assignment, get_model


def main(experiment_parameters, args):
    """ Main function: initializes the dataset and creates benchmark for
        continual learning. Then, it loops over all experiences and trains/evaluates
        the model using a defined strategy.
    """
    # initialize payment dataset
    payment_ds = PaymentDatasetPhiladephia(experiment_parameters['data_dir'])

    # get model
    model = get_model(experiment_parameters, payment_ds.payments_encoded)
    model.eval()

    # create an instance-incremental benchmark by which new samples become available over time
    params = load_params(experiment_parameters["params_path"])
    exp_assignments = get_exp_assignment(params, payment_ds)
    benchmark = ni_benchmark(payment_ds,
                             payment_ds,
                             n_experiences=params["n_experiences"],
                             fixed_exp_assignment=exp_assignments)

    # initialize WandB
    run_name = args.run_name

    # create folder for the current experiment
    output_path = os.path.join(args.outputs_path, run_name)

    sns.color_palette("tab10")
    sns.set_theme()

    # iterate through all experiences (tasks) and train the model over each experience
    for exp_id in range(params["n_experiences"]):
        if exp_id < len(benchmark.train_stream)-1:
            continue
        ckpt_path = os.path.join(output_path, f"ckpt_{exp_id}.pt")
        model.load_state_dict(torch.load(ckpt_path))

        # get experience's dataset
        experience_i = benchmark.train_stream[exp_id]
        main_test_ds_i = copy.copy(experience_i.dataset)

        # number of samples per dept
        num_samples_per_dept = 200

        all_embs = []
        all_labels = []
        for dept_id in params["dept_ids"]:
            if dept_id == 2000:
                continue
            dept_indices = torch.where(main_test_ds_i[:][3] == dept_id)
            if dept_indices == 0:
                continue
            subexp_ds = AvalancheSubset(main_test_ds_i, indices=dept_indices[0])
            embs = model(subexp_ds[:][0], return_z=True)[0].detach()
            all_embs.append(embs)
            all_labels.extend([f'dep {dept_id}']*embs.shape[0])

        all_embs = torch.cat(all_embs, dim=0).cpu().numpy()
        data = {"x": all_embs[:, 0], "y": all_embs[:, 1], "Dept.": all_labels}


        fig = sns.scatterplot(data=data, x="x", y="y", hue="Dept.")
        plt.savefig(os.path.join(output_path, f"plot_{exp_id}.png"))
        plt.clf()


# define main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deepNadim Experiments')
    parser.add_argument('--dataset', help='', nargs='?', type=str, default='philadelphia')  # chicago, philadelphia
    parser.add_argument('--data_dir', help='', nargs='?', type=str, default='./100_datasets/philadelphia')  # chicago, philadelphia
    parser.add_argument('--encoder_dim', help='', nargs='+', default=[128, 64, 32, 16, 8, 4, 2])
    parser.add_argument('--decoder_dim', help='', nargs='+', default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--architecture', help='', nargs='?', type=str, default='baseline')
    parser.add_argument('--architecture_size', help='', nargs='?', type=str, default='small')
    parser.add_argument('--bottleneck', help='', nargs='?', type=str, default='linear')
    parser.add_argument('--params_path', help='', nargs='?', type=str, default='params/params.yml')
    parser.add_argument('--outputs_path', help='', nargs='?', type=str, default='./outputs')
    parser.add_argument('--run_name', help='', nargs='?', type=str, default='')
    args = parser.parse_args()

    experiment_parameter = vars(parser.parse_args())

    # init utilities handler
    uha = UtilsHandler.UtilsHandler()

    # parse string args as int
    if args.architecture_size == "small":
        experiment_parameter['encoder_dim'] = [128, 64, 32, 16, 8, 4, 2]
        experiment_parameter['decoder_dim'] = [2, 4, 8, 16, 32, 64, 128]
    elif args.architecture_size == "large":
        experiment_parameter['encoder_dim'] = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        experiment_parameter['decoder_dim'] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        raise NotImplementedError

    main(experiment_parameter, args)
