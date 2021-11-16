import torch
from avalanche.training.strategies import Naive, EWC, LwF, SynapticIntelligence
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.logging import InteractiveLogger
import NetworkHandler.BaselineAutoencoder as BaselineAutoencoder


class StrategyHandler(object):

    def __init__(self):
        pass

    def get_model(self, experiment_parameter, transactions_encoded):
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

    def get_strategy(self, experiment_parameters, payment_ds):
        # initialize evaluator for metrics and loggers
        interactive_logger = InteractiveLogger()
        eval_plugin = EvaluationPlugin(
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[interactive_logger]
        )

        # initialize model
        model = self.get_model(experiment_parameters, payment_ds.payments_encoded)

        # initialize optimizer
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=experiment_parameters['learning_rate'],
                                     weight_decay=experiment_parameters['weight_decay'])

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
