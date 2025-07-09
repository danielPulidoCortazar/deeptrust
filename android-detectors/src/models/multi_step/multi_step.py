
"""
Python file containing the deeptrust model built within the IIIA.
"""
import logging
import os
import random
import sys
import torch
import omegaconf
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest
from torch.nn.functional import sigmoid, embedding
from torcheval.metrics import BinaryConfusionMatrix
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.base.base_drebin import BaseDREBIN
from models.robust_mlp.robust_mlp import RobustMLP
import torch.nn as nn
from sklearn.utils._array_api import get_namespace
import dill as pkl

class MultiStep(nn.Module, BaseDREBIN):
    """
    Base class for a multi-layer perceptron model.

    Parameters
    ----------
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

        # Set hyperparameters
        self.t1 = cfg["multi_step"].get("t1", None)
        self.t2 = cfg["multi_step"].get("t2", None)
        self.t3 = cfg["multi_step"].get("t3", None)
        self.t4 = cfg["multi_step"].get("t4", None)

        # Load configuration trust_net
        path = cfg["trust_net"].get("cfg_path", None)
        self.trust_net_config = path if os.path.exists(path) else None
        path = cfg["trust_net"].get("classifier_path", None)
        self.trust_net_classifier_path = path if os.path.exists(path) else None
        path = cfg["trust_net"].get("vectorizer_path", None)
        self.trust_net_vectorizer_path = path if os.path.exists(path) else None

        # Load configuration guard_net
        path = cfg["guard_net"].get("cfg_path", None)
        self.guard_net_config = path if os.path.exists(path) else None
        path = cfg["guard_net"].get("classifier_path", None)
        self.guard_net_classifier_path = path if os.path.exists(path) else None
        path = cfg["guard_net"].get("vectorizer_path", None)
        self.guard_net_vectorizer_path = path if os.path.exists(path) else None

        # Load the trust and guard networks
        print("Loading trustNet")
        self.trustNet = RobustMLP.load(self.trust_net_config,
                                       self.trust_net_vectorizer_path,
                                       self.trust_net_classifier_path)
        print("Loading GuardNet")
        self.guardNet = RobustMLP.load(self.guard_net_config,
                                       self.guard_net_vectorizer_path,
                                       self.guard_net_classifier_path)

        # Load the inspectRF
        if self.t3 != 0.0 and self.t3 is not None and self.t3 != "???":
            self.inspectRF = IsolationForest(
                n_estimators=100, max_samples=1.0,
                random_state=0, contamination=self.t3)
        else:
            self.inspectRF = None

        # Set metaheuristic scheme (Multi-step or ensemble)
        self.multi_step = cfg["multi_step"].get("multi_step", False)


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

        if self.inspectRF is not None:
            self.trustNet._load_pt_dataset(X, y, self.trustNet.distillation)

            # Load the PyTorch dataset
            trainloader = self.trustNet.trainloader

            embeddings = []

            self.trustNet.eval()
            with torch.no_grad():
                for i, (features, labels, hard_labels) in enumerate(tqdm(
                        trainloader, desc="Extracting goodware embeddings from baseNet")):

                    # Get the features and labels
                    features = features.to(self.device)
                    hard_labels = hard_labels.to(self.device).squeeze()

                    # Get only goodware samples
                    features = features[hard_labels == 0,:]

                    features = features.to(self.trustNet.device)
                    outputs, batch_embeddings = self.trustNet.forward(
                        features, return_embedding=True)

                    # Append to list
                    embeddings.append(batch_embeddings.cpu())

            # Convert list of tensors to a single tensor
            embeddings = torch.cat(embeddings, dim=0).numpy()

            # Fit the outlier detector inspectRF
            print("Fitting the inspectRF with goodware embeddings.")
            self.inspectRF.fit(embeddings)
        else:
            print("inspectRF is None, skipping fitting.")


    def predict(self, features, ensemble=False):
        """
        Predicts the labels for the given features.

        Parameters
        ----------
        features : CArray
            Features.
        ensemble : bool, optional
            Whether to use ensemble prediction instead the multi-step process.

        Returns
        -------
        CArray
            Predicted labels.
        """

        X = self.trustNet._vectorizer.transform(features)

        if (hasattr(self.trustNet, 'used_features') and
                self.trustNet.used_features is not None):
            X = X[:, self.trustNet.used_features]

        # Impose the batch size to be 1 to facilitate the workflow
        temp = self.trustNet.batch_size
        self.trustNet.batch_size = 1
        dataloader, _ = self.trustNet._load_pt_dataset(X)
        self.trustNet.batch_size = temp

        xp, _ = get_namespace(X)

        scores = np.zeros(len(self.trustNet.set))
        indices = np.zeros(len(self.trustNet.set))

        self.trustNet.eval()
        self.guardNet.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():

                if not self.multi_step:
                    features = batch.to(self.device)
                    guard_output = self.guardNet.forward(features)
                    guard_prob = sigmoid(guard_output).squeeze().cpu().numpy()
                    trust_output = self.trustNet.forward(features)
                    trust_prob = sigmoid(trust_output).squeeze().cpu().numpy()

                    # Combine the probabilities
                    combined_prob = (guard_prob + trust_prob) / 2.0
                    if combined_prob >= self.t1:
                        indices[i] = 1
                        scores[i] = combined_prob

                else:
                    # Step 1: GuardNet
                    features = batch.to(self.device)
                    output = self.guardNet.forward(features)
                    guard_prob = sigmoid(output).squeeze().cpu().numpy()
    #                print("Guard prob: ", guard_prob)
                    if guard_prob >= self.t1:
    #                    print("Guard barrier not crossed!")
                        indices[i] = 1
                        scores[i] = guard_prob

                    # Step 2: BaseNet
                    else:
    #                     print("Guard prob is less than t1")
                        features = batch.to(self.device)
                        output, embedding = self.trustNet.forward(features,
                                                                  return_embedding=True)
                        base_prob = sigmoid(output).squeeze().cpu().numpy()
    #                     print("Base prob: ", base_prob)
                        if base_prob >= self.t2:
    #                         print("Guard barrier crossed! → BaseNet flags it as malware.")
                            indices[i] = 1
                            scores[i] = base_prob

                        # Step 3: InspectorRF
                        elif self.inspectRF is not None:
    #                         print("Base prob is less than t2")
                            np_embedding = embedding.cpu().numpy()
                            is_inlier = self.inspectRF.predict(np_embedding)[0]

                            if is_inlier == 1:
    #                             print("Is inlier and hence is goodware")
                                indices[i] = 0
                                scores[i] = base_prob

                            # Step 4: Goodware outlier classification
                            # with GuardNet
                            else:
    #                             print("Guard barrier crossed! → BaseNet deceived! → Inspector flags it as anomalous → GuardNet takes over.")
                                if guard_prob >= self.t4:
    #                                 print("Guard prob is geq than t4 and hence is malware")
                                    indices[i] = 1
                                else:
    #                                 print("Guard prob is less than t4 and hence is goodware")
                                    indices[i] = 0
                                scores[i] = guard_prob

                        # Go to Step 4 directly if inspectRF is None
                        else:
                            if guard_prob >= self.t4:
                                # Step 4: GuardNet
                                indices[i] = 1
                                scores[i] = guard_prob
                            else:
                                indices[i] = 0
                                scores[i] = base_prob


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

        temp_vectorizer = self._vectorizer
        self._vectorizer = None
        temp_rf = self.inspectRF
        self.inspectRF = None

        # Detach the model from the device
        # Detach model weights from the device
        self.to("cpu")
        torch.save(self.state_dict(), classifier_path)

        self.inspectRF = temp_rf
        with open(classifier_path.replace("ckpt", "pkl"), "wb") as f:
            pkl.dump(self.inspectRF, f)

        self._vectorizer = temp_vectorizer


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

        model = MultiStep(cfg=config_path)
        model._vectorizer = vectorizer
        model._input_features = (model._vectorizer.get_feature_names_out()
                                .tolist())
        # Set device automatically
        # (cuda, if available, else mps, if available, else cpu)
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
        weights = torch.load(
            classifier_path, map_location=device, weights_only=True)
        model.load_state_dict(weights)

        with open(classifier_path.replace("ckpt","pkl"), "rb") as f:
            model.inspectRF = pkl.load(f)

        return model


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