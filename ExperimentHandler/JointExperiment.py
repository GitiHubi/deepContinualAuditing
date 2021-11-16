import numpy as np
import torch
import wandb
import copy
import os
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
from DataHandler.PaymentDataset import PaymentDataset
from UtilsHandler.UtilsHandler import UtilsHandler
from UtilsHandler.StrategyHandler import StrategyHandler
from UtilsHandler.BenchmarkHandler import BenchmarkHandler


def run_joint_experiment(experiment_parameters):
    # Initialize handlers
    uha = UtilsHandler()
    sha = StrategyHandler()
    bha = BenchmarkHandler()

    # Initialize payment dataset
    payment_ds = PaymentDataset(experiment_parameters['data_dir'])

    # Get index assignments for all experiences
    exp_assignments = bha.get_exp_assignment(experiment_parameters, payment_ds)

    # Get benchmark
    benchmark = bha.get_benchmark(experiment_parameters, payment_ds, exp_assignments)

    # Get Strategy
    strategy = sha.get_strategy(experiment_parameters, payment_ds)

    # Initialize WandB
    run_name = experiment_parameters['run_name']
    log_wandb = experiment_parameters['wandb_proj'] != ''
    uha.init_wandb(experiment_parameters, run_name, log_wandb)
    output_path = os.path.join(experiment_parameters['outputs_path'], run_name)

    # ============================
    # Train jointly and evaluate on all experiences
    # ============================

    res_train = strategy.train(benchmark.train_stream)
    loss_train_exp = res_train[f"Loss_Epoch/train_phase/train_stream/Task000"]

    # eval loss for all experiences
    last_exp_id = len(benchmark.train_stream) - 1
    res_eval = strategy.eval(benchmark.train_stream)
    loss_eval_exp = res_eval[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{last_exp_id:03d}"]

    if log_wandb:
        wandb.log({"experience/loss_train": loss_train_exp}, step=last_exp_id)
        wandb.log({f"experience/loss_exp_{last_exp_id}": loss_eval_exp}, step=last_exp_id)

    # ============================
    # Compute per-department losses in the final experience
    # ============================

    loss_per_dep = [[] for i in experiment_parameters["dept_ids"]]
    exp_i = benchmark.train_stream[last_exp_id]
    main_test_ds_i = copy.copy(exp_i.dataset)
    for itr_dep, dept_id in enumerate(experiment_parameters["dept_ids"]):
        dept_indices = torch.where(main_test_ds_i[:][3] == dept_id)
        subexp_ds = AvalancheSubset(main_test_ds_i, indices=dept_indices[0])
        if len(subexp_ds) > 0:
            exp_i.dataset = subexp_ds
            res = strategy.eval(exp_i)
            loss_dept_i = res[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{last_exp_id:03d}"]
        else:
            loss_dept_i = None
        loss_per_dep[itr_dep].append(loss_dept_i)

    if log_wandb:
        for itr_dep, dept_id in enumerate(experiment_parameters["dept_ids"]):
            dep_losses = loss_per_dep[itr_dep]
            if dep_losses[-1] != None:
                wandb.log({f"dept/loss_dept{dept_id}": dep_losses[-1]}, step=last_exp_id)

        torch.save(strategy.model.state_dict(), os.path.join(output_path, f"ckpt_0.pt"))
        wandb.finish()
