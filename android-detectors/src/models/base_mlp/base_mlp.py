"""
Python file containing the base iiia_mlp model built within the IIIA.
"""
import os
import random
import sys
import torch
from scipy.sparse import csr_matrix
from torch.nn.functional import sigmoid
import numpy as np
import omegaconf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from iiia_data import PtDrebinDataset
from models.base.base_drebin import BaseDREBIN
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils._array_api import get_namespace
from sklearn.model_selection import train_test_split
import dill as pkl


class BaseMLP(nn.Module, BaseDREBIN):
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

        nn.Module.__init__(self)
        BaseDREBIN.__init__(self)

        # Set device automatically (cuda, if available, else mps, if available, else cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        if type(cfg) == omegaconf.dictconfig.DictConfig:
            # If cfg is a DictConfig, convert it to a dictionary
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        else:
            # Parse the configuration file
            assert os.path.exists(cfg) or "Configuration file not found."
            cfg = omegaconf.OmegaConf.load(cfg)
            # To dictionary
            cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)

        # Get hyperparameters
        assert cfg.get("trainer", None) is not None or "Trainer configs not declared."
        self.batch_size = cfg["trainer"].get('batch_size', 32)
        self.patience = cfg["trainer"].get('patience', 3)
        self.min_epochs = cfg["trainer"].get('min_epochs', 3)
        self.max_epochs = cfg["trainer"].get('max_epochs', 10)
        self.lr_rate= cfg["trainer"].get('lr_rate', 0.001)
        self.optimizer = cfg["trainer"].get('optimizer', 'adam')
        self.optimizer_params = cfg["trainer"].get('optimizer_params', {'weight_decay': 0.002463768595899745})
        self.loss = cfg["trainer"].get('loss', 'bce')
        self.loss_params = cfg["trainer"].get('loss_params', {'pos_weight': 8.5})

        assert cfg.get("model", None) is not None or "Model configs not declared."
        self.in_dim = cfg["model"].get('in_dim', 1461078)
        self.out_dim = cfg["model"].get('out_dim', 1)
        self.hidden_sizes = cfg["model"].get('hidden_sizes', [256, 32, 256])
        self.activation = cfg["model"].get('activation', "leaky_relu")
        self.dropout = cfg["model"].get('dropout', 0.7000000000000001)

        # Create the linear layers
        ## Create the input layer
        self.input_layer = nn.Linear(self.in_dim, self.hidden_sizes[0]).to(self.device)
        # Print the weights of the input layer
        print(f"Reproducibility check: \n"
                 f"Input layer weights - {self.input_layer.weight}")
        ## Create the hidden layers with dropout
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.to(self.device)
        ## Create the output layer
        self.output_layer = nn.Linear(self.hidden_sizes[-1], 1)
        self.output_layer.to(self.device)

        # Create activation function
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # Create the optimizer
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_rate, **self.optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Create the loss function
        if self.loss == 'bce':
            if self.loss_params.get('pos_weight', None):
                # If not converted to dict, it throws error
                dict_loss_params = dict(self.loss_params)
                dict_loss_params['pos_weight'] = torch.tensor(dict_loss_params['pos_weight'])
                dict_loss_params['pos_weight'] = dict_loss_params['pos_weight'].to(self.device)

            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum",
                **dict_loss_params)

        else:
            raise ValueError(f"Unknown loss function: {self.loss}")


    def forward(self, x, return_embedding=False):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        return_embedding : bool, optional
            Whether to return the output of the last hidden layer.

        Returns
        -------
        torch.Tensor
            Output tensor.
        torch.Tensor, optional
            Embedding tensor if return_embedding is True.
        """

        x = self.input_layer(x)

        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_fn(x)

        # The embedding is the final activation before the output layer
        embedding = x

        x = self.output_layer(x)

        return (x, embedding) if return_embedding else x


    def _load_pt_dataset(self, X: csr_matrix, y = None, distillation=0.0):
        """
        Loads the dataset into a PyTorch dataset.

        Parameters
        ----------
        X : CArray
            Features.
        y : CArray
            Labels.
        distillation : float, optional
            Value between 0 and 1.

        Returns
        -------
        DataLoader
            The PyTorch DataLoader.
        DataLoader (optional)
            The PyTorch DataLoader.
        """

        # Create the PyTorch dataset
        self.set = PtDrebinDataset(X, y, distillation)

        # If the labels are provided, split the dataset for training
        if self.set.y is not None:
            # Split in a stratified way the dataset into train and validation (80-20)
            train_idx, val_idx = train_test_split(
                np.arange(len(self.set)), test_size=0.2, stratify=self.set.hard_labels)
            self.valset = torch.utils.data.Subset(self.set, val_idx)
            self.trainset = torch.utils.data.Subset(self.set, train_idx)
            print(f"Reproducibility check: First 10 train split idx -  {train_idx[:10]}")
            print(f"Reproducibility check: First 10 val split idx - {val_idx[:10]}")

            # Create the dataloaders
            self.trainloader = DataLoader(
                self.trainset, batch_size=self.batch_size, shuffle=True,num_workers=0)
            self.valloader = DataLoader(
                self.valset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            return self.trainloader, self.valloader

        else:
            self.dataloader = DataLoader(
                self.set, batch_size=self.batch_size, shuffle=False)

            return self.dataloader, None


    def _compute_metrics(self, loss, tp, fp, tn, fn, i):
        """
        Computes the metrics.

        Parameters
        ----------
        loss : torch.Tensor
            The loss.
        tp : int
            True positives.
        fp : int
            False positives.
        tn : int
            True negatives.
        fn : int
            False negatives.
        i : int
            The index of the batch for the mean loss computation.

        Returns
        -------
        dict
            The metrics.
        """

        acc = np.round((tp + tn) / (tp + fp + tn + fn), 4).item()
        prec = np.round(tp / (tp + fp), 4).item()
        rec = np.round(tp / (tp + fn), 4).item()
        spec = np.round(tn / (tn + fp), 4).item()
        f1_score = np.round(2 * (prec * rec) / (prec + rec), 4).item() if prec + rec > 0 else torch.nan
        mean_loss = np.round(loss.detach().cpu() / (self.batch_size * (i + 1)), 6).item()
        fp_rate = np.round(fp / (fp + tn), 4).item() if fp + tn > 0 else torch.nan
        metrics_dict = {'L': mean_loss, 'A': acc, 'P': prec, 'R': rec, 'S': spec,
                        'F1': f1_score, 'FPR': fp_rate}

        return metrics_dict


    def _fit(self, X, y):
        """
        Trains the model.
        """

        raise NotImplementedError


    def predict(self, features):
        """
        Predicts the labels for the given features.

        Parameters
        ----------
        features : CArray
            Features.

        Returns
        -------
        CArray
            Predicted labels.
        CArray
            Predicted scores.
        """

        X = self._vectorizer.transform(features)

        if (hasattr(self, 'used_features') and
                self.used_features is not None):
            X = X[:, self.used_features]

        dataloader, _ = self._load_pt_dataset(X)
        xp, _ = get_namespace(X)

        scores = np.zeros(len(self.set))
        self.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                features = batch.to(self.device)
                outputs = self.forward(features)
                probs = sigmoid(outputs).squeeze().cpu().numpy().tolist()
                scores[i * self.batch_size: (i + 1) * self.batch_size] = probs

        indices = xp.astype(scores >= 0.5, dtype=int)

        return indices, scores


    def save(self, vectorizer_path, classifier_path):
        """
        Saves the model.

        Parameters
        ----------
        vectorizer_path : str
            Path to save the vectorizer.
        classifier_path : str
            Path to save the classifier.
        """

        with open(vectorizer_path, "wb") as f:
            pkl.dump(self._vectorizer, f)

        vectorizer = self._vectorizer
        self._vectorizer = None
        torch.save(self.state_dict(), classifier_path)
        self._vectorizer = vectorizer


    @staticmethod
    def load(vectorizer_path, classifier_path):
        """
        Loads the model.

        Parameters
        ----------
        vectorizer_path : str
            Path to load the vectorizer.
        classifier_path : str
            Path to load the classifier.

        Returns
        -------
        MLP
            The loaded model.
        """

        raise NotImplementedError


    def fit(self, features, y, feat_selection=np.empty(0)):
        """
        Parameters
        ----------
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
        y : np.ndarray
            Array of shape (n_samples,) containing the class labels.
        """

        X = self._vectorizer.fit_transform(features)
        self._input_features = (self._vectorizer.get_feature_names_out()
                                .tolist())

        self.used_features = None
        if feat_selection.any():
            self.used_features = feat_selection
            self._input_features = [self._input_features[idx] for idx in self.used_features]
            X = X[:, self.used_features]

        train_metrics = self._fit(X, y)

        return train_metrics