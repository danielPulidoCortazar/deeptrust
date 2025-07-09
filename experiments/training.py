import json
import os

from omegaconf import OmegaConf

from models import RobustMLP, NaturalMLP, MultiStep
from models.utils import load_features, load_labels

def run_experiment(config, dir_name, model="natural"):
    """
    Run the training experiment with the given configuration and save the model and results.

    Parameters
    ----------
    config : OmegaConf
        Configuration object containing the training parameters.
    dir_name : str
        Directory name to save the model and results.
    model : str
        Type of model to train. Options are "natural", "robust", or "multistep".
    """
    # Load data
    features_tr = load_features("data/training_set_features.zip")
    y_tr = load_labels("data/training_set_features.zip",
                       "data/training_set.zip")

    # Initialize the classifier
    if model == "natural":
        classifier = NaturalMLP(cfg=config)
    elif model == "robust":
        classifier = RobustMLP(cfg=config)
    elif model =="multistep":
        classifier = MultiStep(cfg=config)
    else:
        raise ValueError("Invalid model type. Choose 'natural', 'robust', or 'multi_step'.")

    # Train the classifier
    train_results = classifier.fit(features_tr, y_tr)

    # Create directories for saving model and results
    path = f"experiments/out/{dir_name}"
    if os.path.exists(path):
        print(f"The directory {path} already exists. Files will be overwritten.")
    os.makedirs(path, exist_ok=True)

    # Save the model
    print("Saving the model to disk in: ", path)
    classifier.save(vectorizer_path=f"{path}/vectorizer.pkl",
                    classifier_path=f"{path}/classifier.ckpt")

    # Save the config to yaml
    with open(f"{path}/config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Save the results
    with open(f"{path}/train_metrics.json", "w") as f:
        json.dump(train_results, f)


if __name__ == "__main__":

    # NATURAL TRAINING EXPERIMENTATION

    # Configuration for natural training
    print("========================================")
    print("Starting natural training experimentation...")
    print("========================================")
    print("Experimental setup:")
    print("- Natural training with predefined model and trainer parameters")
    print("========================================")

    # Load the configuration file
    config_path = "experiments/configs/mlp_training.yaml"
    config = OmegaConf.load(config_path)

    # Run the experiment with natural training
    run_experiment(config, dir_name="natural", model="natural")

    print(f"Training completed for experiment: natural-train")
    print("=========================================")
    print()

 # =================================================================================
 # =================================================================================

     # TABULAR ADVERSARIAL TRAINING EXPERIMENTATION

    # # Configurations for tabular adversarial training experimentation
    # m_configs = [1, 2, 5, 10, 15, 20]
    # delta_bound_configs = [25, 50, 75, 100]
    # feat_selection_configs = ["topk", "random"]
    #
    # print("========================================")
    # print("Starting tabular adversarial training experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Adversarial training experimentation with:")
    # print(f"- m_configs: {m_configs}")
    # print(f"- delta_bound_configs: {delta_bound_configs}")
    # print(f"- feat_selection_configs: {feat_selection_configs}")
    # print("========================================")
    # for m in m_configs:
    #     for delta_bound in delta_bound_configs:
    #         for feat_selection in feat_selection_configs:
    #             # Print the current configuration
    #             print(f"Training with m={m}, delta_bound={delta_bound}, "
    #                   f"feat_selection={feat_selection}")
    #
    #             # Load the configuration file
    #             config_path = "experiments/configs/mlp_training.yaml"
    #             config = OmegaConf.load(config_path)
    #             config.adversarial_trainer.m = m
    #             config.adversarial_trainer.delta_bound = delta_bound
    #             config.adversarial_trainer.feat_selection = feat_selection
    #             config.distillation.distillation = 0.0
    #
    #             run_experiment(
    #                 config,
    #                 f"tab-adv-train-m_{m}_delta_bound_{delta_bound}_feat_selection_{feat_selection}",
    #                 model="robust"
    #             )
    #
    #             print(f"Training completed for experiment: tab-adv-train with m={m}, delta_bound={delta_bound}, "
    #                   f"feat_selection={feat_selection}")
    #             print("=========================================")
    #             print()

 # =================================================================================
 # =================================================================================

    # ADVERSARIAL DISTILLATION EXPERIMENTATION

    # # Configurations for adversarial distillation configurations
    # distillation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #
    # print("========================================")
    # print("Starting adversarial distillation experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Adversarial distillation experimentation with:")
    # print(f"- distillation: {distillation}")
    # print("========================================")
    # for d in distillation:
    #     # Print the current configuration
    #     print(f"Training with distillation={d}")
    #
    #     # Load the configuration file
    #     config_path = "experiments/configs/mlp_training.yaml"
    #     config = OmegaConf.load(config_path)
    #     config.adversarial_trainer.m = 1
    #     config.adversarial_trainer.delta_bound = 0
    #     config.adversarial_trainer.feat_selection = "topk"
    #     config.distillation.distillation = d
    #
    #     run_experiment(
    #         config,
    #         f"adv-distill-train-distillation_{d}",
    #         model="robust"
    #     )
    #
    #     print(f"Training completed for experiment: adv-distill-train with distillation={d}")
    #     print("=========================================")
    #     print()

# =================================================================================
# =================================================================================

    # MULTISTEP EXPERIMENTATION with {best adv train mlp -> natural mlp -> best tab adv train mlp}

    # # Configurations for multistep training
    # t3_configs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #
    # print("========================================")
    # print("Starting multistep training experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Multistep using diverging models, training experimentation with:")
    # print(f"- t3_configs: {t3_configs}")
    # print("========================================")
    # for t3 in t3_configs:
    #     # Print the current configuration
    #     print(f"Training with t3={t3}")
    #
    #     # Load the configuration file
    #     config_path = "experiments/configs/multistep_training.yaml"
    #     config = OmegaConf.load(config_path)
    #     config.multi_step.t3 = t3
    #     config.guard_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/config.yaml"
    #     config.guard_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/classifier.ckpt"
    #     config.guard_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/vectorizer.pkl"
    #     config.trust_net.cfg_path = "experiments/out/natural-train/config.yaml"
    #     config.trust_net.classifier_path = "experiments/out/natural-train/classifier.ckpt"
    #     config.trust_net.vectorizer_path = "experiments/out/natural-train/vectorizer.pkl"
    #
    #     run_experiment(
    #         config,
    #         f"multistep-divergent-t3_{t3}",
    #         model="multistep"
    #     )
    #
    #     print(f"Training completed for experiment: multistep-divergent with t3={t3}")
    #     print("=========================================")
    #     print()

# =================================================================================
# =================================================================================

    # MULTISTEP EXPERIMENTATION with {best adv train mlp -> 2nd best adv train mlp -> best tab adv train mlp}

    # # Configurations for multistep training
    # t3_configs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #
    # print("========================================")
    # print("Starting multistep training experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Multistep using converging models, training experimentation with:")
    # print(f"- t3_configs: {t3_configs}")
    # print("========================================")
    # for t3 in t3_configs:
    #     # Print the current configuration
    #     print(f"Training with t3={t3}")
    #
    #     # Load the configuration file
    #     config_path = "experiments/configs/multistep_training.yaml"
    #     config = OmegaConf.load(config_path)
    #     config.multi_step.t3 = t3
    #     config.guard_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/config.yaml"
    #     config.guard_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/classifier.ckpt"
    #     config.guard_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/vectorizer.pkl"
    #     config.trust_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/config.yaml"
    #     config.trust_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/classifier.ckpt"
    #     config.trust_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/vectorizer.pkl"
    #
    #     run_experiment(
    #         config,
    #         f"multistep-convergent-t3_{t3}",
    #         model="multistep"
    #     )
    #
    #     print(f"Training completed for experiment: multistep-convergent with t3={t3}")
    #     print("=========================================")
    #     print()

    # =================================================================================
    # =================================================================================

    # ENSEMBLE EXPERIMENTATION with {best adv train mlp + natural mlp}

    # print("========================================")
    # print("Starting ensemble training experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Ensemble using diverging models")
    # print("========================================")
    #
    # # Load the configuration file
    # config_path = "experiments/configs/multistep_training.yaml"
    # config = OmegaConf.load(config_path)
    # config.multi_step.multi_step = False # Turn into an ensemble when doing inference
    # config.multi_step.t1 = 0.5
    # config.guard_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/config.yaml"
    # config.guard_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/classifier.ckpt"
    # config.guard_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/vectorizer.pkl"
    # config.trust_net.cfg_path = "experiments/out/natural-train/config.yaml"
    # config.trust_net.classifier_path = "experiments/out/natural-train/classifier.ckpt"
    # config.trust_net.vectorizer_path = "experiments/out/natural-train/vectorizer.pkl"
    #
    # run_experiment(
    #     config,
    #     f"ensemble-divergent",
    #     model="multistep"
    # )
    #
    # print(f"Training completed for experiment: ensemble-divergent")
    # print("=========================================")
    # print()

    # =================================================================================
    # =================================================================================

    # ENSEMBLE EXPERIMENTATION with {best adv train mlp + 2nd best adv train mlp}

    # print("========================================")
    # print("Starting ensemble training experimentation...")
    # print("========================================")
    # print("Experimental setup:")
    # print("Predefined model and trainer parameters")
    # print(f"Ensemble using converging models")
    # print("========================================")
    #
    # # Load the configuration file
    # config_path = "experiments/configs/multistep_training.yaml"
    # config = OmegaConf.load(config_path)
    # config.multi_step.multi_step = False # Turn into an ensemble when doing inference
    # config.multi_step.t1 = 0.5
    # config.guard_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/config.yaml"
    # config.guard_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/classifier.ckpt"
    # config.guard_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_75_feat_selection_topk/vectorizer.pkl"
    # config.trust_net.cfg_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/config.yaml"
    # config.trust_net.classifier_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/classifier.ckpt"
    # config.trust_net.vectorizer_path = "experiments/out/tab-adv-train-m_2_delta_bound_100_feat_selection_topk/vectorizer.pkl"
    #
    # run_experiment(
    #     config,
    #     f"ensemble-convergent",
    #     model="multistep"
    # )
    #
    # print(f"Training completed for experiment: ensemble-convergent")
    # print("=========================================")
    # print()

    # =================================================================================
    # =================================================================================

print("All experiments completed.")