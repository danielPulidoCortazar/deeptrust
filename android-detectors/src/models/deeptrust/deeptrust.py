
"""
Python file containing the deeptrust model built within the IIIA.
"""
import logging
import os
import random
import sys
import torch
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest
from torch.nn.functional import sigmoid, embedding
from torcheval.metrics import BinaryConfusionMatrix
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.base.base_drebin import BaseDREBIN
from models.trust_mlp.trust_mlp import TrustMLP
from models.guard_mlp.guard_mlp import GuardMLP
import torch.nn as nn
from sklearn.utils._array_api import get_namespace
import dill as pkl

class DeepTrust(nn.Module, BaseDREBIN):
    """
    Base class for a multi-layer perceptron model.

    Parameters
    ----------
    """

    def __init__(self):
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

        # Set hyperparameters
        self.h1 =  0.78
        self.h2 = 0.5
        self.h3 = 0.14
        self.h4 = 0.5

        # Load configuration trustnet
        path = 'android-detectors/pretrained/trustnet_classifier.pkl'
        self.trustnet_classifier_path = path if os.path.exists(path) else None
        path = 'android-detectors/pretrained/trustnet_vectorizer.pkl'
        self.trustnet_vectorizer_path = path if os.path.exists(path) else None

        # Load configuration guardnet
        path = 'android-detectors/pretrained/guardnet_classifier.pkl'
        self.guardnet_classifier_path = path if os.path.exists(path) else None
        path = 'android-detectors/pretrained/guardnet_vectorizer.pkl'
        self.guardnet_vectorizer_path = path if os.path.exists(path) else None

        # Load the base and guard networks
        if (self.trustnet_classifier_path is None and self.trustnet_vectorizer_path is None):
            print("Initializing trustNet")
            self.trustNet = TrustMLP()
        else:
            print("Loading trustNet")
            self.trustNet = TrustMLP.load(self.trustnet_vectorizer_path, self.trustnet_classifier_path)

        if  (self.trustnet_classifier_path is None and self.trustnet_vectorizer_path is None):
            print("Initializing guardNet")
            self.guardNet = GuardMLP()
        else:
            print("Loading guardNet")
            self.guardNet = GuardMLP.load(self.guardnet_vectorizer_path, self.guardnet_classifier_path)

        # True for multistep, False for averaging the probs. of the two networks
        self.multistep = True

        # Load the inspectRF
        self.inspectRF = IsolationForest(
            n_estimators=100, max_samples=1.0,
            random_state=0, contamination=self.h3)


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
            self.trustNet.load_pt_dataset(X, y, self.trustNet.distillation)

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
            print("No inspectRF provided, skipping the fitting of the outlier detector.")


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
        """

        X = self.trustNet._vectorizer.transform(features)

        if (hasattr(self.trustNet, 'used_features') and
                self.trustNet.used_features is not None):
            X = X[:, self.trustNet.used_features]

        # Impose the batch size to be 1 to facilitate the workflow
        temp = self.trustNet.batch_size
        self.trustNet.batch_size = 1
        dataloader, _ = self.trustNet.load_pt_dataset(X)
        self.trustNet.batch_size = temp

        xp, _ = get_namespace(X)

        scores = np.zeros(len(self.trustNet.set))
        indices = np.zeros(len(self.trustNet.set))

        self.trustNet.eval()
        self.guardNet.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():

                if not self.multistep:
                    features = batch.to(self.device)
                    guard_output = self.guardNet.forward(features)
                    guard_prob = sigmoid(guard_output).squeeze().cpu().numpy()
                    trust_output = self.trustNet.forward(features)
                    trust_prob = sigmoid(trust_output).squeeze().cpu().numpy()

                    # Combine the probabilities
                    combined_prob = (guard_prob + trust_prob) / 2.0
                    if combined_prob >= self.h4:
                        indices[i] = 1
                        scores[i] = combined_prob

                else:
                    # Step 1: GuardNet
                    features = batch.to(self.device)
                    output = self.guardNet.forward(features)
                    guard_prob = sigmoid(output).squeeze().cpu().numpy()

                    if guard_prob >= self.h1:
                        indices[i] = 1
                        scores[i] = guard_prob

                    # Step 2: BaseNet
                    else:
                        features = batch.to(self.device)
                        output, embedding = self.trustNet(features, return_embedding=True)
                        base_prob = sigmoid(output).squeeze().cpu().numpy()

                        # Check 1: Is BaseNet confident it is Malware?
                        is_malware = base_prob >= self.h2

                        # Check 2: Is it a verified Inlier (Goodware) via InspectorRF?
                        # We only check this if it's NOT predicted as malware already.
                        is_inlier = False
                        if not is_malware and self.inspectRF is not None:
                            np_embedding = embedding.cpu().numpy()
                            # Check if RF predicts class 1 (Inlier)
                            is_inlier = self.inspectRF.predict(np_embedding)[0]

                        # Final Assignment
                        if is_malware:
                            # Case A: BaseNet says Malware
                            indices[i] = 1
                            scores[i] = base_prob

                        elif is_inlier == 1:
                            # Case B: InspectorRF says Goodware
                            indices[i] = 0
                            scores[i] = base_prob

                        else:
                            # Case C: Fallback to GuardNet
                            # Occurs if:
                            # 1. BaseNet was unsure (< h2) AND
                            # 2. (InspectorRF was missing OR InspectorRF flagged it as an outlier)
                            indices[i] = 1 if guard_prob >= self.h4 else 0
                            scores[i] = guard_prob

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
        torch.save(self.state_dict(), classifier_path.replace("pkl", "pth"))

        self.inspectRF = temp_rf
        with open(classifier_path, "wb") as f:
            pkl.dump(self.inspectRF, f)

        self._vectorizer = temp_vectorizer


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
        with open(vectorizer_path, "rb") as f:
            vectorizer = pkl.load(f)

        model = DeepTrust()
        model._vectorizer = vectorizer
        model._input_features = (model._vectorizer.get_feature_names_out()
                                .tolist())
        # Set device automatically
        # (cuda, if available, else mps, if available, else cpu)
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
        weights = torch.load(
            classifier_path.replace("pkl","pth"), map_location=device, weights_only=True)
        model.load_state_dict(weights)

        with open(classifier_path, "rb") as f:
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