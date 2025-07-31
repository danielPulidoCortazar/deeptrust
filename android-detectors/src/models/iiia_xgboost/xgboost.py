import logging
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils._array_api import get_namespace
from models.base import BaseDREBIN
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix as BinaryConfusionMatrix


log = logging.getLogger(__name__)

class XGBoost(BaseDREBIN, XGBClassifier):
    """
    Implements the XGBoost classifier using xgboost library.
    Documentation for pararemeters: https://xgboost.readthedocs.io/en/stable/parameter.html.
    """

    def __init__(self,
                 n_estimators=100, max_depth=8, max_leaves=None,
                 learning_rate=0.2, gamma=0.25, subsample=1.0, colsample_bytree=1.0,
                 colsample_bylevel=1.0):
        """
        Default hyparameters are the optimal ones found with TPE hyperparameter optimization.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the ensemble. Default is 100.
        max_depth : int, optional
            Maximum depth of a tree. Default is 8. Increasing this value will make the
            model more complex and more likely to overfit.
        max_leaves : int, optional
            Maximum number of leaves in a tree. Default is None.
        learning_rate : float, optional
            Learning rate shrinks the contribution of each tree. Default is 0.2.
        gamma : float, optional
            Minimum loss reduction required to make a further partition on a leaf node. Default is 0.
            The larger the value, the more conservative the algorithm will be.
        subsample : float, optional
            Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would
            randomly sample half of the training data prior to growing trees. and this will
            prevent overfitting. Subsampling will occur once in every boosting iteration. range: (0,1]
            Default is 1.0.
        colsample_bytree : float, optional
            Subsample ratio of columns when constructing each tree. Default is 1.0.
        colsample_bylevel : float, optional
            Subsample ratio of columns for each level of the tree. Default is 1.0.
        """


        # Set seeds
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        BaseDREBIN.__init__(self)
        XGBClassifier.__init__(self, n_estimators=n_estimators, max_depth=max_depth, max_leaves=max_leaves,
                      grow_policy='lossguide', learning_rate=learning_rate, verbosity=2,
                      objective='binary:logistic', booster='gbtree', tree_method="exact", gamma=gamma,
                      subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                      random_state=0)

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

        XGBClassifier.fit(self, X[train_idx], y[train_idx])
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
