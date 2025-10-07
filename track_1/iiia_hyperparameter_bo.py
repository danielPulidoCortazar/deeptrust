"""
Bayesian optimization for hyperparameter tuning of the models.
"""
import sys
sys.path.append("../android-detectors/src")

import logging
import math
import random

import hydra
import numpy as np
import plotly
import torch
from models import NaturalMLP, RobustMLP, MultiStep, RF, XGBoost
import optuna
from models.utils import *
import os
from omegaconf import OmegaConf

from track_1.iiia_evaluate import iiia_evaluate

# Set logging
logging.basicConfig(level=logging.INFO)
# Redirect Optuna logger to Hydra logger
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.INFO)  # Adjust the level as needed
optuna_logger.propagate = True

def objective_mlp(trial):

    # Set seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_mlp(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)
    logging.info(f"Metrics: {train_metrics}")
    f1 = np.nanmax(train_metrics["val_f1"])
    score = f1 if not math.isnan(f1) else 0.0

    return score

def robust_objective_mlp(trial):

    # Set seeds: Very important to set seeds in the objective function!
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_mlp(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)

    metrics = iiia_evaluate(classifier)

    # Geometric mean of fpr, adv_25_recall, adv_50_recall, adv_100_recall
    fpos_acc = metrics["fpos_metrics"]["acc"]
    pos_acc = metrics["pos_metrics"]["acc"]
    adv_25_acc = metrics["adv_metrics_25"]["acc"]
    adv_50_acc = metrics["adv_metrics_50"]["acc"]
    adv_100_acc = metrics["adv_metrics_100"]["acc"]
    logging.info(f"Metrics: {metrics}")
    # To modulate the penalization when fpos_acc is below 0.99. If fpos_acc is below 0.95, the score is 0.
    fpr_modulation = max(0, (0.05 - (1 - fpos_acc)) / 0.05)
    score = fpr_modulation * (pos_acc * adv_25_acc * adv_50_acc * adv_100_acc) ** (1/4)

    return score

def robust_objective_advmlp(trial):

    # Set seeds: Very important to set seeds in the objective function!
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_advmlp(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)

    metrics = iiia_evaluate(classifier)

    # Geometric mean of fpr, adv_25_recall, adv_50_recall, adv_100_recall
    fpos_acc = metrics["fpos_metrics"]["acc"]
    pos_acc = metrics["pos_metrics"]["acc"]
    adv_25_acc = metrics["adv_metrics_25"]["acc"]
    adv_50_acc = metrics["adv_metrics_50"]["acc"]
    adv_100_acc = metrics["adv_metrics_100"]["acc"]
    logging.info(f"Metrics: {metrics}")
    # To modulate the penalization when fpos_acc is below 0.99. If fpos_acc is below 0.95, the score is 0.
    fpr_modulation = max(0, (0.05 - (1 - fpos_acc)) / 0.05)
    score = fpr_modulation * ((pos_acc * adv_25_acc * adv_50_acc * adv_100_acc) ** (1/4))

    return score

def robust_objective_multistep(trial):

    # Set seeds: Very important to set seeds in the objective function!
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_duomlp(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)

    metrics = iiia_evaluate(classifier)

    # Geometric mean of fpr, adv_25_recall, adv_50_recall, adv_100_recall
    fpos_acc = metrics["fpos_metrics"]["acc"]
    #pos_acc = metrics["pos_metrics"]["acc"]
    #adv_25_acc = metrics["adv_metrics_25"]["acc"]
    #adv_50_acc = metrics["adv_metrics_50"]["acc"]
    adv_100_acc = metrics["adv_metrics_100"]["acc"]
    logging.info(f"Metrics: {metrics}")
    # To modulate the penalization when fpos_acc is below 0.99. If fpos_acc is below 0.95, the score is 0.
    #fpr_modulation = max(0, (0.03 - (1 - fpos_acc)) / 0.03)
    fpr_modulation = max(0, 1 - max(0, 0.99 - fpos_acc))
    score = fpr_modulation * (adv_100_acc)

    return score

def objective_rf(trial):

    # Set seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_rf(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)
    logging.info(f"Metrics: {train_metrics}")
    f1 = np.nanmax(train_metrics["val_f1"])
    score = f1 if not math.isnan(f1) else 0.0

    return score

def objective_xgboost(trial):

    # Set seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    logging.info(f"Trial number: {trial.number}")

    # Define the model
    classifier = define_trial_xgboost(trial)

    base_path = os.path.join(os.path.dirname(__file__))

    features_tr = load_features(
        os.path.join(base_path, "../data/training_set_features.zip"))
    y_tr = load_labels(
        os.path.join(base_path, "../data/training_set_features.zip"),
        os.path.join(base_path, "../data/training_set.zip"))

    train_metrics = classifier.fit(features_tr, y_tr)
    logging.info(f"Metrics: {train_metrics}")
    f1 = np.nanmax(train_metrics["val_f1"])
    score = f1 if not math.isnan(f1) else 0.0

    return score

def define_trial_mlp(trial):

    # Define the hyperparameters to optimize
    hidden_sizes = []

    # Define num_neurons for hidden layer 0
    hidden_0_exp = trial.suggest_int(
            f"hidden_0_exp", 7, 8) # 128, 256
    hidden_size_0 = 2 ** hidden_0_exp
    # Define num_neurons for hidden layer 1
    hidden_1_exp = trial.suggest_int(
            f"hidden_1_exp", 5, 8)  # 32, 64, 128, 256
    # Define num_neurons for hidden layer 2
    hidden_2_exp = trial.suggest_int(
            f"hidden_2_exp", 5, 8)  # 32, 64, 128, 256

    for i in range(3):
        if i == 0:
            hidden_sizes.append(hidden_size_0)
        elif i == 1:
            hidden_sizes.append(2 ** hidden_1_exp)
        elif i == 2 and hidden_2_exp > 0:
            hidden_sizes.append(2 ** hidden_2_exp)

    model = {
        "in_dim": 1461078,
        "out_dim": 1,
        "hidden_sizes": hidden_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "leaky_relu"]),
        "dropout": trial.suggest_float("dropout", 0, 0.75, step=0.05),
    }
    trainer = {
        "batch_size": 32,
        "patience": 3,
        "min_epochs": 3,
        "max_epochs": 10,
        "lr_rate": 0.001,
        "optimizer": "adam",
        "optimizer_params": {"weight_decay":
                                 trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
                             },
        "loss": "bce",
        "loss_params": {'pos_weight': trial.suggest_float("pos_weight", 1, 9, step=0.5)},
    }

    # Define the model
    cfg = {"model": model, "trainer": trainer}
    cfg = OmegaConf.create(cfg)
    logging.info(f"Trial configuration:\n{OmegaConf.to_yaml(cfg)}")

    model = NaturalMLP(cfg)

    return model

def define_trial_advmlp(trial):

    adversarial_trainer = {
        "perturbation_scheme": trial.suggest_categorical("pert_scheme", ['reset', "accumulate"]),
        "m": trial.suggest_int("m", 2, 20, step=2),
        "classes_to_perturb": [1],
        "delta_type": "discrete", #"continuous"
        "delta_bound": trial.suggest_int("delta_bound", 25, 200, step=25), # trial.suggest_float("delta_bound", 0.01, 0.5, step=0.01)
        "feat_selection": trial.suggest_categorical("feat_selection", ["topk", "random"]),
    }
    model = {
        "in_dim": 1461078,
        "out_dim": 1,
        "hidden_sizes": [256, 32, 256],
        "activation": "leaky_relu",
        "dropout": 0.7000000000000001
    }
    trainer = {
        "batch_size": 32,
        "patience": 3,
        "min_epochs": 3,
        "max_epochs": trial.suggest_int("max_epochs", max(adversarial_trainer["m"],10), 20),
        "lr_rate": 0.001,
        "optimizer": "adam",
        "optimizer_params": {"weight_decay": 0.002463768595899745},
        "loss": "bce",
        "loss_params": {'pos_weight': 8.5}
    }
    distillation = {"distillation": trial.suggest_float("distillation", 0.0, 0.5, step=0.05)}

    # Define the model
    cfg = {"model": model, "trainer": trainer,
           "adversarial_trainer": adversarial_trainer,
           "distillation": distillation}
    cfg = OmegaConf.create(cfg)
    logging.info(f"Trial configuration:\n{OmegaConf.to_yaml(cfg)}")
    model = RobustMLP(cfg)

    return model

def define_trial_multistep(trial):

    multi_step = {
        "multi_step": True, # Use multi-step training, otherwise output fusion using t1
        't1': trial.suggest_float("t1", 0.75, 0.90, step=0.01),  # Thresh. to classify as malware when guardNet prob is high
        't2': trial.suggest_float("t2", 0.45, 0.55, step=0.01), # Thresh. for baseNet decision
        't3': trial.suggest_float("t3", 0.05, 0.2, step=0.01),  # Contamination rate for fitting inspectorRF
        't4': trial.suggest_float("t4", 0.45, 0.55, step=0.01), # Thresh. for guardNet decision
    }
    trust_net = {
    'cfg_path': '',
    'classifier_path': '',
    'vectorizer_path': '',
    }
    guard_net = {
    'cfg_path': '',
    'classifier_path': '',
    'vectorizer_path': '',
    }

    # Define the model
    cfg = {"multi_step": multi_step, 'trust_net': trust_net, 'guard_net': guard_net}
    cfg = OmegaConf.create(cfg)
    logging.info(f"Trial configuration:\n{OmegaConf.to_yaml(cfg)}")
    model = MultiStep(cfg)

    return model

def define_trial_rf(trial):

    hyperparameters = {
        "n_estimators": trial.suggest_int("n_estimators", 25, 100, step=5),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": None,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 502, step=50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 501, step=50),
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": False,
        "oob_score": False,
        "n_jobs": None,
        "random_state": 0,
        "verbose": 0,
        "warm_start": False,
        "class_weight": trial.suggest_categorical("class_weight",
                                                  [None, "balanced", "balanced_subsample"]),
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    }

    # Define the model
    cfg = {"hyperparameters": hyperparameters}
    cfg = OmegaConf.create(cfg)

    n_estimators = cfg.hyperparameters.n_estimators
    # Pop hyperparameters from cfg
    hyper = dict(cfg.hyperparameters)
    hyper.pop("n_estimators")
    logging.info(f"Trial configuration:\n{OmegaConf.to_yaml(cfg)}")
    model = RF(n_estimators=n_estimators, **hyper)

    return model

def define_trial_xgboost(trial):

    hyperparameters = {
        "n_estimators": trial.suggest_int("n_estimators", 25, 100, step=5),
        "max_depth": trial.suggest_int("max_depth", 3, 10, step=1),
        "max_leaves": None,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0, step=0.01),
        "subsample": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.05),
        "colsample_bylevel": 1,
    }

    # Define the model
    cfg = {"hyperparameters": hyperparameters}
    cfg = OmegaConf.create(cfg)

    n_estimators = cfg.hyperparameters.n_estimators
    # Pop hyperparameters from cfg
    hyper = dict(cfg.hyperparameters)
    hyper.pop("n_estimators")
    logging.info(f"Trial configuration:\n{OmegaConf.to_yaml(cfg)}")
    model = XGBoost(n_estimators=n_estimators, **hyper)

    return model


@hydra.main(version_base="1.3.2", config_path="iiia_config", config_name="bo_config")
def main(cfg):

    # Print formatted configuration
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if cfg.model == "mlp":
        study.optimize(robust_objective_mlp, n_trials=20, n_jobs=1)
    elif cfg.model == "advmlp":
        study.optimize(robust_objective_advmlp, n_trials=20, n_jobs=1)
    elif cfg.model == "multistep":
        study.optimize(robust_objective_multistep, n_trials=10, n_jobs=1)
    elif cfg.model == "rf":
        study.optimize(objective_rf, n_trials=20, n_jobs=1)
    elif cfg.model == "xgboost":
        study.optimize(objective_xgboost, n_trials=20, n_jobs=1)

    logging.info("Finished search.")

    logging.info(f"Study statistics: \n"
                 f"  Number of finished trials: {len(study.trials)}")

    trial = study.best_trial
    logging.info(f"Best trial value: "
                    f"  Value: {trial.value:.4f}\n"
                    f"  Params: {trial.params}")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_parallel_coordinate.html")

    fig = optuna.visualization.plot_optimization_history(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_history.html")

    fig = optuna.visualization.plot_slice(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_slice.html")

    fig = optuna.visualization.plot_contour(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_contour.html")

    fig = optuna.visualization.plot_param_importances(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_importances.html")

    fig = optuna.visualization.plot_edf(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_edf.html")

    fig = optuna.visualization.plot_rank(study)
    plotly.io.write_html(fig, cfg.run_dir + "/opt_rank.html")

    logging.info("Optimization results saved.")

if __name__ == "__main__":

    main()