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


class RobustMLP(BaseMLP, BaseDREBIN):
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

        # Set train mode hyperparameters
        assert cfg.get("adversarial_trainer", None) is not None or "Adversarial trainer configs not declared."
        self.perturbation_scheme = cfg["adversarial_trainer"].get('perturbation_scheme', 'accumulate')
        self.m = cfg["adversarial_trainer"].get('m', 1)
        self.classes_to_perturb = cfg["adversarial_trainer"].get('classes_to_perturb', [1])
        self.delta_bound = cfg["adversarial_trainer"].get('delta_bound', 0)
        self.delta_type = cfg["adversarial_trainer"].get('delta_type', 'discrete')
        self.feat_selection = cfg["adversarial_trainer"].get('feat_selection', 'topk')

        # Get distillation params
        assert cfg.get("distillation", None) is not None or "Distillation configs not declared."
        self.distillation = 0.0
        if cfg.get("distillation", None) is not None:
            self.distillation = cfg["distillation"].get('distillation', 0.0)

        # Get smoothing params
        assert cfg.get("smoothing", None) is not None or "Smoothing configs not declared."
        self.smoothing = 0.0
        if cfg.get("smoothing", None) is not None:
            self.smoothing = cfg["smoothing"].get('smoothing', 0.0)


    def _fit(self, X, y):
        """
        Fit the model with tabular adversarial training algorithm evolved from
        (Shafahi et al., 2019).

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
        self._load_pt_dataset(X, y, distillation=self.distillation, smoothing=self.smoothing)

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

        # Initialize delta to vector of zeros
        if self.perturbation_scheme == 'accumulate':
            delta_global = torch.zeros(
                self.batch_size, self.in_dim, requires_grad=False, device='mps')

        for epoch in range(1, (self.max_epochs // self.m) + 1):
            with tqdm(iterable=self.trainloader, total=len(self.trainloader),
                      desc= f'Epoch {epoch}, train', postfix=metrics_dict) as pbar:
                train_loss = 0.0
                self.train()

                for i, (features, labels, hard_labels) in enumerate(self.trainloader):
                    # Get the features and labels
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    if self.perturbation_scheme == 'reset':
                        delta_global = torch.zeros(
                            self.batch_size, self.in_dim, requires_grad=False, device='mps')

                    for _ in range(self.m):
                        # Clone to reset the gradients
                        perturbed_features = features.detach().clone()

                        # Get indices of the samples to perturb
                        indices_to_perturb = torch.tensor(
                            [i for i in range(self.batch_size)
                             if labels[i].item() in self.classes_to_perturb])

                        # Perturb samples
                        if len(indices_to_perturb) > 0:
                            if self.delta_type == 'discrete':
                                perturbed_features[indices_to_perturb] = torch.clamp(
                                    perturbed_features[indices_to_perturb] + delta_global[indices_to_perturb],
                                    0, 1)

                            if self.delta_type == 'continuous':
                                perturbed_features[indices_to_perturb] += delta_global[indices_to_perturb]

                        # Forward pass with perturbed samples
                        outputs = self.forward(perturbed_features)

                        # If distillation is enabled, set new loss_fn to use hard labels to set the weights
                        if self.distillation > 0.0:
                            self.loss_fn = nn.BCEWithLogitsLoss(
                                reduction="sum",
                                weight=hard_labels.to(self.device) *
                                       self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                        # Compute the loss
                        loss = self.loss_fn(outputs, labels)
                        train_loss += loss

                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()

                        # Repeat the process to update delta global
                        if len(indices_to_perturb) > 0:
                            # Clone delta_global to reset the gradients
                            delta = delta_global.detach().clone().requires_grad_(True)
                            perturbed_features = features.detach().clone()

                            if self.delta_type == 'discrete':
                                perturbed_features[indices_to_perturb] = torch.clamp(
                                    perturbed_features[indices_to_perturb] + delta[indices_to_perturb],
                                    0, 1)

                            if self.delta_type == 'continuous':
                                perturbed_features[indices_to_perturb] += delta[indices_to_perturb]

                            # Forward pass with perturbed samples
                            outputs = self.forward(perturbed_features)

                            if self.distillation > 0.0:
                                # Create weighted binary cross entropy loss based on the hard labels
                                self.loss_fn = nn.BCEWithLogitsLoss(
                                    reduction="sum",
                                    weight=hard_labels.to(self.device) *
                                           self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                            loss = self.loss_fn(outputs, labels)
                            grad = torch.autograd.grad(loss, delta)[0].detach().cpu()
                            sign_indices = torch.sign(grad)

                            if self.delta_type == 'discrete':

                                if self.feat_selection == 'topk':
                                    # Get k features with the highest gradients
                                    feat_indices = torch.topk(torch.abs(grad), self.delta_bound, dim=1)[1]

                                elif self.feat_selection == 'random':
                                    # Get random k features
                                    feat_indices = torch.randint(0, self.in_dim, (self.batch_size, self.delta_bound))

                                sign_indices = torch.gather(sign_indices, 1, feat_indices)

                                # Update delta_global
                                delta_temp = torch.zeros(self.batch_size, self.in_dim, requires_grad=False,
                                                         device='mps')
                                delta_temp.scatter_(
                                    1, feat_indices.to(self.device), sign_indices.to(self.device))

                                delta_global[indices_to_perturb] += delta_temp[indices_to_perturb]
                                delta_global[indices_to_perturb] = torch.clamp(
                                    delta_global[indices_to_perturb], -1, 1)

                                # For each sample in the batch, scale the number of changes to epsilon
                                for j in indices_to_perturb:
                                    bin_diff = torch.abs(delta_global[j])
                                    diff_indices = torch.nonzero(bin_diff).squeeze()
                                    n_changes = int(bin_diff.sum().item())
                                    if n_changes > self.delta_bound:
                                        indices_perm = torch.randperm(n_changes)
                                        restore_indices = indices_perm[:n_changes - self.delta_bound]
                                        restore_feats = diff_indices[restore_indices]
                                        delta_global[j, restore_feats] = 0

                            elif self.delta_type == 'continuous':
                                # Update delta_global
                                delta_global[indices_to_perturb] += self.delta_bound * sign_indices[indices_to_perturb].to(self.device)

                                # Clip delta_global
                                delta_global[indices_to_perturb] = torch.clamp(
                                    delta_global[indices_to_perturb], -self.delta_bound, self.delta_bound)

                        # Optimize
                        self.optimizer.step()

                        if self.distillation > 0.0 or self.smoothing > 0.0:
                            # Compute cm and metrics
                            cm.update(input=sigmoid(outputs).squeeze(),
                                      target=hard_labels.squeeze().to(torch.int64)
                                      )
                        else:
                            # Compute cm and metrics
                            cm.update(input=sigmoid(outputs).squeeze(),
                                      target=labels.squeeze().to(torch.int64)
                                      )

                        tn, fp, fn, tp = cm.compute().reshape(-1)
                        metrics_dict = self._compute_metrics(train_loss, tp, fp, tn, fn, i)

                        # Update the metrics of the progress bar
                        pbar.set_postfix(metrics_dict)

                    # Update the progress bar
                    pbar.update(1)

                # Store the metrics and reset the cm
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

                        if self.distillation > 0.0:
                            # Create weighted binary cross entropy loss based on the hard labels
                            self.loss_fn = nn.BCEWithLogitsLoss(
                                reduction="sum",
                                weight=hard_labels.to(self.device) *
                                       self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                        # Compute the loss
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss

                    if self.distillation > 0.0:
                        # Compute cm and metrics
                        cm.update(input=sigmoid(outputs).squeeze(),
                                  target=hard_labels.squeeze().to(torch.int64)
                                  )
                    else:
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

                print(f"Epoch {epoch}, val metrics: {metrics_dict}")

            # Reset the confusion matrix
            cm.reset()

        # To avoid conflicts when loading the model
        if self.distillation > 0.0:
            # If not converted to dict, it throws error
            dict_loss_params = dict(self.loss_params)
            dict_loss_params['pos_weight'] = torch.tensor(dict_loss_params['pos_weight'])
            dict_loss_params['pos_weight'] = dict_loss_params['pos_weight'].to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum",
                                                **dict_loss_params)

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

        model = RobustMLP(cfg=config_path)
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