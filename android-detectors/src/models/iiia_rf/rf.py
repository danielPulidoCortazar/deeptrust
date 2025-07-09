import logging
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils._array_api import get_namespace
from models.base import BaseDREBIN
from sklearn.ensemble import RandomForestClassifier

from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix as BinaryConfusionMatrix


log = logging.getLogger(__name__)

class RF(BaseDREBIN, RandomForestClassifier):
    """
    Implements the RandomForest classifier.
    """

    def __init__(self,
                 n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=0,
                 verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                 max_samples=None, monotonic_cst=None):


        # Set seeds
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        BaseDREBIN.__init__(self)
        RandomForestClassifier.__init__(self, n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                        max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                        oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                        verbose=verbose, warm_start=warm_start, class_weight=class_weight,
                                        ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst
        )

    def _compute_metrics(self, tp, fp, tn, fn):
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

        acc = np.round((tp + tn) / (tp + fp + tn + fn), 4)
        prec = np.round(tp / (tp + fp), 4)
        rec = np.round(tp / (tp + fn), 4)
        spec = np.round(tn / (tn + fp), 4)
        f1_score = np.round(2 * (prec * rec) / (prec + rec), 4) if prec + rec > 0 else torch.nan
        fp_rate = np.round(fp / (fp + tn), 4) if fp + tn > 0 else torch.nan
        metrics_dict = {'A': acc, 'P': prec, 'R': rec, 'S': spec,
                        'F1': f1_score, 'FPR': fp_rate}

        return metrics_dict

    def _fit(self, X, y):

        # Split in a stratified way the dataset into train and validation (80-20)
        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y)
        log.info(f"Reproducibility check: First 10 train split idx -  {train_idx[:10]}")
        log.info(f"Reproducibility check: First 10 val split idx - {val_idx[:10]}")

        RandomForestClassifier.fit(self, X[train_idx], y[train_idx])
        # Train metrics
        y_train = y[train_idx]
        X_train = X[train_idx]
        train_scores = self.predict_proba(X_train)
        train_preds = np.argmax(train_scores, axis=1)

        # Initialize the confusion matrix from sklearn and dictionaries
        cm = BinaryConfusionMatrix(y_true=y_train, y_pred=train_preds)
        train_metrics = {
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_spec': [], 'val_spec': [],
            'train_f1': [], 'val_f1': [],
            'train_fpr': [], 'val_fpr': []
        }

        # Compute train metrics with sklearn confusion matrix
        tn, fp, fn, tp = cm.ravel()
        train_acc, train_prec, train_rec, train_spec, train_f1, train_fpr = self._compute_metrics(tp, fp, tn, fn).values()
        train_metrics['train_acc'].append(train_acc)
        train_metrics['train_prec'].append(train_prec)
        train_metrics['train_rec'].append(train_rec)
        train_metrics['train_spec'].append(train_spec)
        train_metrics['train_f1'].append(train_f1)
        train_metrics['train_fpr'].append(train_fpr)

        # Compute validation metrics
        y_val = y[val_idx]
        X_val = X[val_idx]
        val_scores = self.predict_proba(X_val)
        val_preds = np.argmax(val_scores, axis=1)

        # Compute validation metrics with sklearn confusion matrix
        cm = BinaryConfusionMatrix(y_true=y_val, y_pred=val_preds)
        tn, fp, fn, tp = cm.ravel()
        val_acc, val_prec, val_rec, val_spec, val_f1, val_fpr = self._compute_metrics(tp, fp, tn, fn).values()
        train_metrics['val_acc'].append(val_acc)
        train_metrics['val_prec'].append(val_prec)
        train_metrics['val_rec'].append(val_rec)
        train_metrics['val_spec'].append(val_spec)
        train_metrics['val_f1'].append(val_f1)
        train_metrics['val_fpr'].append(val_fpr)

        return train_metrics

    def fit(self, features, y):
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

        train_metrics = self._fit(X, y)

        return train_metrics

    def predict(self, features):
        X = self._vectorizer.transform(features)
        xp, _ = get_namespace(X)
        scores = self.predict_proba(X)
        if len(scores.shape) == 1:
            indices = xp.astype(scores > 0.5, int)
        else:
            indices = xp.argmax(scores, axis=1)

        return xp.take(self.classes_, indices, axis=0), scores[:, 1]
