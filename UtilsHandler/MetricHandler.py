import math
import torch
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset


class MetricHandler(object):

    def __init__(self):
        pass

    def compute_FP_ratio(self, strategy, ds, params):
        """
        Compute FP for department 10.
        :param model: stategy's models
        :param ds: dataset
        :param params: dictionary of parameters
        :return:
        """
        global_anomaly_dept = 1000
        local_anomaly_dept = 2000

        # Remove all global anomalies
        valid_indices = torch.where(ds[:][3] != global_anomaly_dept)[0]
        ds = AvalancheSubset(ds, indices=valid_indices)

        # Compute loss for all dataset entries
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        # Compute reconstruction losses for all samples in the dataset
        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        # Number of all local anomalies
        k = torch.sum(ds[:][3]==local_anomaly_dept).item()
        _, indices = torch.topk(recon_losses, k=10, largest=True)

        # False positives
        fp = 0
        for idx in indices:
            if ds[idx][3].item() in params["target_dept_ids"]:
                fp += 1
        fp_ratio = fp / float(k)

        return fp_ratio

    def compute_FN_ratio(self, strategy, ds, params):
        """
        Computes FN for local anomalies (dep. 2000)
        :param model:
        :param ds:
        :param params: dictionary of parameters
        :return:
        """
        global_anomaly_dept = 1000
        local_anomaly_dept = 2000

        # Remove all global anomalies
        valid_indices = torch.where(ds[:][3] != global_anomaly_dept)[0]
        ds = AvalancheSubset(ds, indices=valid_indices)

        # Compute loss for all dataset entries
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        # Compute reconstruction losses for all samples in the dataset
        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        # Number of all local anomalies
        k = torch.sum(ds[:][3] == local_anomaly_dept).item()
        _, indices = torch.topk(recon_losses, k=10, largest=True)

        # False negatives
        fn = 0
        for idx in indices:
            if ds[idx][3].item() != local_anomaly_dept:
                fn += 1
        fn_ratio = fn / float(k)

        return fn_ratio
