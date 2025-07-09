"""
Python file containing the base iiia_mlp model built within the IIIA.
"""
import json
import os
import random
import sys
import torch
import omegaconf
from torch.nn.functional import sigmoid
from torcheval.metrics import BinaryConfusionMatrix
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from iiia_data import PtDrebinDataset
from models.base.base_drebin import BaseDREBIN
from models.base_mlp.base_mlp import BaseMLP
import torch.nn as nn
import dill as pkl


class NaturalMLP(BaseMLP, BaseDREBIN):
    """
    Base class for a multi-layer perceptron model.
    """

    def __init__(self, cfg=None):
        """
        Initializes the model.
        """

        # Set seeds
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Initialize the base classes
        BaseMLP.__init__(self, cfg=cfg)
        BaseDREBIN.__init__(self)

        if type(cfg) == omegaconf.dictconfig.DictConfig:
            # If cfg is a DictConfig, convert it to a dictionary
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        else:
            # Parse the configuration file
            assert os.path.exists(cfg) or "Configuration file not found."
            cfg = omegaconf.OmegaConf.load(cfg)
            # To dictionary
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

        # Print the configuration
        print(f"Configuration:")
        print(json.dumps(cfg, indent=4))


    def _fit(self, X, y):
        """
        Fits the model.

        Parameters
        ----------
        X : CArray
            Features.
        y : CArray
            Labels.

        Returns
        -------
        dict
            The training metrics.
        """

        # Load the PyTorch dataset
        self._load_pt_dataset(X, y)

        # Show num of params
        print(f"Number of trainable parameters: "
                 f"{sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

        # Initialize the confusion matrix and dictionaries
        cm = BinaryConfusionMatrix(threshold=0.5)
        metrics_dict = {}
        train_metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_spec': [], 'val_spec': [],
            'train_f1': [], 'val_f1': [],
            'train_fpr': [], 'val_fpr': []
        }

        patience = self.patience
        best_model = self.state_dict()

        for epoch in range(1, self.max_epochs + 1):
            with tqdm(iterable=self.trainloader, total=len(self.trainloader),
                      desc=f'Epoch {epoch}, train', postfix=metrics_dict) as pbar:
                train_loss = 0.0
                self.train()

                for i, (features, labels, hard_labels) in enumerate(self.trainloader):
                    # Zero the gradients
                    self.optimizer.zero_grad()
                    # Get the features and labels
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    # Forward pass
                    outputs = self.forward(features)

                    # Compute the loss
                    loss = self.loss_fn(outputs, labels)
                    train_loss += loss
                    # Backward pass
                    loss.backward()
                    # Optimize
                    self.optimizer.step()

                    # Compute cm and metrics
                    cm.update(input=sigmoid(outputs).squeeze(),
                              target=labels.squeeze().to(torch.int64)
                              )
                    tn, fp, fn, tp = cm.compute().reshape(-1)
                    metrics_dict = self._compute_metrics(train_loss, tp, fp, tn, fn, i)

                    # Update the progress bar
                    pbar.set_postfix(metrics_dict)
                    pbar.update(1)

            # Store the metrics
            train_metrics['train_loss'].append(metrics_dict['L'])
            train_metrics['train_acc'].append(metrics_dict['A'])
            train_metrics['train_prec'].append(metrics_dict['P'])
            train_metrics['train_rec'].append(metrics_dict['R'])
            train_metrics['train_spec'].append(metrics_dict['S'])
            train_metrics['train_f1'].append(metrics_dict['F1'])
            train_metrics['train_fpr'].append(metrics_dict['FPR'])

            print(f"Epoch {epoch}, train metrics: {metrics_dict}")

            # Reset the confusion matrix
            cm.reset()

            with tqdm(iterable=self.valloader, total=len(self.valloader),
                      desc=f'Epoch {epoch}, val', postfix=metrics_dict) as pbar:

                self.eval()
                val_loss = 0.0

                for i, (features, labels, hard_labels) in enumerate(self.valloader):
                    with torch.no_grad():
                        # Get the features and labels
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        # Forward pass
                        outputs = self.forward(features)

                        # Compute the loss
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss

                    # Compute cm and metrics
                    cm.update(input=sigmoid(outputs).squeeze(),
                              target=labels.squeeze().to(torch.int64)
                              )
                    tn, fp, fn, tp = cm.compute().reshape(-1)
                    metrics_dict = self._compute_metrics(val_loss, tp, fp, tn, fn, i)

                    # Update the progress bar
                    pbar.set_postfix(metrics_dict)
                    pbar.update(1)

            # Store the metrics and reset the cm
            train_metrics['val_loss'].append(metrics_dict['L'])
            train_metrics['val_acc'].append(metrics_dict['A'])
            train_metrics['val_prec'].append(metrics_dict['P'])
            train_metrics['val_rec'].append(metrics_dict['R'])
            train_metrics['val_spec'].append(metrics_dict['S'])
            train_metrics['val_f1'].append(metrics_dict['F1'])
            train_metrics['val_fpr'].append(metrics_dict['FPR'])

            print(f"Epoch {epoch}, patience: {patience}, val metrics: {metrics_dict}")

            cm.reset()

            # Early stopping
            if (self.min_epochs < epoch and epoch > 1
                    and train_metrics['val_f1'][-1] < np.max(train_metrics['val_f1'][self.min_epochs - 1:-1])):
                patience -= 1
                if patience == 0:
                    break
            else:
                # Save the model
                print("Saving model at epoch %d" % epoch)
                best_model = self.state_dict()
                patience = self.patience

        if len(train_metrics['val_f1']) > 0:
            # Load the best model
            best_epoch = np.argmax(train_metrics['val_f1'][self.min_epochs - 1:]) + self.min_epochs
            print("Loading the best model from epoch %d based on the validation F1 score" % best_epoch)
            self.load_state_dict(best_model)

        return train_metrics


    @staticmethod
    def load(config_path, vectorizer_path, classifier_path):
        """
        Loads the model.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        vectorizer_path : str
            Path to load the vectorizer.
        classifier_path : str
            Path to load the classifier.

        Returns
        -------
        MLP
            The loaded model.
        """
        with open(vectorizer_path, "rb") as f:
            vectorizer = pkl.load(f)

        model = NaturalMLP(cfg=config_path)
        model._vectorizer = vectorizer
        model._input_features = (model._vectorizer.get_feature_names_out()
                                .tolist())

        if classifier_path is not None:
            # Set device automatically
            # (cuda, if available, else mps, if available, else cpu)
            device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else
                                  "cpu")
            weights = torch.load(
                classifier_path, map_location=device, weights_only=True)
            model.load_state_dict(weights)

        return model