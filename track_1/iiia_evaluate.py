import numpy as np
from sklearn.metrics import confusion_matrix

from feature_space_attack import FeatureSpaceAttack
from models.utils import *
import os
import logging



def _compute_metrics(y_true, y_pred):
    """
    Compute the confusion matrix and return the accuracy, precision, recall,
    specificity, F1 score, and false positive rate.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray

    Returns
    -------
    acc : float
        Accuracy.
    prec : float
        Precision.
    rec : float
        Recall.
    spec : float
        Specificity.
    f1_score : float
        F1 score.
    fp_rate : float
        False positive rate.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    acc = np.nan if np.sum(cm) == 0 else np.round((tp + tn) / np.sum(cm), 4)
    prec = np.nan if tp + fp == 0 else np.round(tp / (tp + fp), 4)
    rec = np.nan if tp + fn == 0 else np.round(tp / (tp + fn), 4)
    spec = np.nan if tn + fp == 0 else np.round(tn / (tn + fp), 4)
    f1_score = np.nan if np.isnan(prec) or np.isnan(rec) or prec + rec == 0 else \
        np.round(2 * prec * rec / (prec + rec), 4)
    fp_rate = np.nan if fp + tn == 0 else np.round(fp / (fp + tn), 4)

    return acc, prec, rec, spec, f1_score, fp_rate

def iiia_evaluate(classifier):
    """
    Evaluate the classifier on the test set.

    Parameters
    ----------
    classifier : models.MLP
        The classifier.

    Returns
    -------
    metrics : dict
        Dictionary containing the evaluation
        metrics.
    """

    base_path = os.path.join(os.path.dirname(__file__))

    ## Evaluate the classifier on the set of goodware samples
    features_fp_check = load_features(
            os.path.join(base_path, "../data/test_set_fp_check_features.zip"))
    y_pred, scores = classifier.predict(features_fp_check)
    gt_labels = np.zeros(len(y_pred))
    # Compute metrics
    acc, prec, rec, spec, f1_score, fp_rate = _compute_metrics(gt_labels, y_pred)
    fp_metrics = {'fpos_metrics': {'acc': acc, 'prec': prec, 'rec': rec,
     'spec': spec, 'f1': f1_score, 'fpr': fp_rate}}
    logging.log.info(f"Metrics on the fp set: {fp_metrics}")

    ## Evaluate the classifier on the set of malware samples
    malware_features = load_features(
        os.path.join(base_path, "../data/test_set_adv_features.zip"))
    y_pred, scores = classifier.predict(malware_features)
    gt_labels = np.ones(len(y_pred))
    # Compute metrics
    acc, prec, rec, spec, f1_score, fp_rate = _compute_metrics(gt_labels, y_pred)
    pos_metrics = {'pos_metrics': {'acc': acc, 'prec': prec, 'rec': rec,
     'spec': spec, 'f1': f1_score, 'fpr': fp_rate}}
    log.info(f"Metrics on the pos set: {pos_metrics}")

    ## Evaluate the classifier on the set of adversarial samples
    attack = FeatureSpaceAttack(classifier=classifier,
                                logging_level=logging.DEBUG)
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    # Get random subset of malware samples using a generator class
    class MalwareFeatureGenerator:
        def __init__(self, features, indices):
            self.features = features
            self.indices = set(indices)  # Convert to set for faster lookup

        def __iter__(self):
            for i, sample in enumerate(self.features):
                if i in self.indices:
                    yield sample

    # Randomly select 50 indices
    indices = np.random.choice(len(gt_labels), 50, replace=False)

    adv_metrics = {}
    for n_feats in [100]: # [25, 50, 100]:

        logging.log.info(f"Running attack with a maximum of {n_feats} feature changes")

        goodware_features = (
            sample for sample, label in zip(load_features(
            os.path.join(base_path, "../data/training_set_features.zip")),
            y_tr) if label == 0)
        malware_features = load_features(
            os.path.join(base_path, "../data/test_set_adv_features.zip"))

        logging.log.info("Reproducibility check: "
                 "First 10 indices of the random malware subset: %s" % indices[:10])
        # Convert malware_features to a generator class
        malware_features = MalwareFeatureGenerator(malware_features, indices)

        # Generate adversarial examples
        adv_examples = attack.run(
            malware_features, goodware_features, n_iterations=100,
            n_features=n_feats, n_candidates=50)

        y_pred, scores = classifier.predict(adv_examples)
        gt_labels = np.ones(len(y_pred))

        # Compute metrics
        acc, prec, rec, spec, f1_score, fp_rate = _compute_metrics(gt_labels, y_pred)
        adv_metrics[f'adv_metrics_{n_feats}'] = {'acc': acc, 'prec': prec, 'rec': rec,
            'spec': spec, 'f1': f1_score, 'fpr': fp_rate}
        logging.log.info(f"Metrics on the adv set with {n_feats} feature changes: {adv_metrics}")

        metrics = {**fp_metrics, **pos_metrics, **adv_metrics}

    return metrics