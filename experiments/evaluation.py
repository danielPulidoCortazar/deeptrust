import argparse
import json
import multiprocessing as mp
import os
import time

import sys

from omegaconf import omegaconf

import config
from track_1.evaluation import evaluate

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "android-detectors", "src"))
from models.utils import *
from models import RobustMLP, MultiStep

def evaluate_model(model_dir):
    # Get paths
    config_path = os.path.join(model_dir, "config.yaml")
    vect_path = os.path.join(model_dir, "vectorizer.pkl")
    clf_path = os.path.join(model_dir, "classifier.ckpt")

    cfg = omegaconf.OmegaConf.load(config_path)
    if not cfg.get("multi_step", False):
        # Load the pre-trained model
        classifier = RobustMLP.load(config_path, vect_path, clf_path)
    else:
        # Load the pre-trained model
        classifier = MultiStep.load(config_path, vect_path, clf_path)


    # Evaluate the model
    results = evaluate(classifier, config)

    # Add to the results the name of the model
    # Get the last part of the path
    model_name = model_dir.split("/")[-1]

    results = [{"model": model_name}] + results

    # Create directories for saving model and results
    dir_name = f"{model_dir.split('/')[-1]}"
    path = f"experiments/out/{dir_name}"
    os.makedirs(path, exist_ok=True)
    # Save in json
    with open(f"{path}/evaluation.json", "w") as f:
        json.dump(results, f)

    return results


def parallel_evaluation(model_dirs):
    # Check if the model directories are empty
    if not model_dirs:
        print("No model directories found. Exiting.")
        return []

    # Check if the model directories are valid
    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            print(f"Invalid model directory: {model_dir}. Exiting.")
            return []

    # Check if the model directories contain the required files
    for model_dir in model_dirs:
        if not os.path.isfile(os.path.join(model_dir, "config.yaml")):
            print(f"Missing config.yaml in model directory: {model_dir}. Exiting.")
            return []
        if not os.path.isfile(os.path.join(model_dir, "vectorizer.pkl")):
            print(f"Missing vectorizer.pkl in model directory: {model_dir}. Exiting.")
            return []
        if not os.path.isfile(os.path.join(model_dir, "classifier.ckpt")):
            print(f"Missing classifier.ckpt in model directory: {model_dir}. Exiting.")
            return []

    # Parallel Processing
    print("Starting parallel evaluation...")
    pool = mp.Pool(processes=mp.cpu_count())
    time_start = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(evaluate_model, model_dirs)

    time_end = time.time()
    print(f"Time taken: {round(time_end - time_start,4)} seconds")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--models_dir",
        type=str,
        default="experiments/out",
        help="Path to the model directories to evaluate"
    )
    args = arg_parser.parse_args()
    models_dir = args.models_dir

    # Check if the dir is a model directory
    if (os.path.exists(models_dir+"/config.yaml") or
            os.path.exists(models_dir+"/vectorizer.pkl") or
            os.path.exists(models_dir+"/classifier.ckpt")):
        print("The directory is a model directory. Evaluating the model...")
        evaluate_model(models_dir)

    else:
        print("Detected multiple model directories in the provided path.")
        # Get all the model directories
        model_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d))]
        parallel_evaluation(model_dirs)