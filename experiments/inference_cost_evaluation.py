import argparse
import json
import os
import sys
import time
import numpy as np
import tracemalloc
from tqdm import tqdm

# --- Project Path Setup ---
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "android-detectors", "src"))
import config
from models.utils import *
from models import DeepTrust, RF, XGBoost, GuardMLP, TrustMLP, DREBIN, SecSVM


def _measure_dataset(dataset_name, features_path, model, runs_per_sample=1) -> dict:
    print(f"\n--- Processing {dataset_name} ---")

    # 1. Load Data (Outside Timer)
    print("Loading features...")
    features_gen = load_features(features_path)
    features_list = list(features_gen)
    n_samples = len(features_list)

    sample_latencies = []

    # 2. Iterate Samples
    iterator = tqdm(range(n_samples), desc=f"Inferencing {dataset_name}")

    for i in iterator:
        raw_sample = features_list[i]

        # 3. Pre-process (Outside Timer)
        if hasattr(raw_sample, "toarray"):
            sample = raw_sample.toarray()
        elif hasattr(raw_sample, "reshape"):
            sample = raw_sample.reshape(1, -1)
        else:
            sample = np.array(raw_sample).reshape(1, -1)

        # 4. Measure Inference
        start_time = time.perf_counter()
        for _ in range(runs_per_sample):
            model.predict(sample)
        end_time = time.perf_counter()

        # Avg time per sample (ms)
        avg_ms = ((end_time - start_time) / runs_per_sample) * 1000
        sample_latencies.append(avg_ms)

    # 5. Compile Results
    return {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "avg_latency_per_sample_ms": round(np.mean(sample_latencies), 4),
        "total_latency_ms": round(np.sum(sample_latencies), 4)
    }


def evaluate_inference_cost(model_loader_func, model_name="DeepTrust", runs_per_sample=1) -> list[dict]:
    """
    Evaluates the inference cost (Memory + Latency) of a model.

    Parameters
    ----------
    model_loader_func : function
        A function that loads and returns the model to be evaluated.
    model_name : str, optional
        The name of the model (for reporting purposes), by default "DeepTrust".
    runs_per_sample : int, optional
        The number of times to run inference per sample for averaging latency, by default 1.

    Returns
    -------
    results : list of dict
        A list of dictionaries containing the evaluation results, including memory usage and latency metrics.
    """

    print(f"[{model_name}] Starting Evaluation...")
    results = []

    # --- 1. Measure Memory (RAM) ---
    print("Measuring Memory Footprint...")

    # Start tracing memory allocations
    tracemalloc.start()

    # Load the model
    model = model_loader_func()

    # Snapshot memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model_size_mb = current / (1024 * 1024)
    peak_load_mb = peak / (1024 * 1024)

    print(f"Model Size (RAM): {model_size_mb:.2f} MB")
    print(f"Peak Loading RAM: {peak_load_mb:.2f} MB")

    results.append(
        {"model_size_ram_mb": round(model_size_mb, 2),
         "peak_loading_ram_mb": round(peak_load_mb, 2)}
    )

    # --- 2. Run Goodware ---
    gw_res = _measure_dataset("Goodware (FP Check)",
                              config.FEATURES_TS_FP_CHECK,
                              model,
                              runs_per_sample)
    results.append(gw_res)

    # --- 3. Run Malware ---
    mw_res = _measure_dataset("Malware",
                              config.FEATURES_TS_ADV,
                              model,
                              runs_per_sample)
    results.append(mw_res)

    # --- 4. Save Results ---
    output_dir = "experiments/out/inference_cost"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/cost_evaluation_{model_name}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate inference cost for specific models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Name of the model to evaluate (e.g., DeepTrust, RandomForest, XGBoost). Use 'all' to run everything."
    )
    args = parser.parse_args()

    # We define them here so they are available to be called dynamically
    def load_deeptrust():
        vect_path = "android-detectors/pretrained/deeptrust_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/deeptrust_classifier.pkl"
        model = DeepTrust.load(vect_path, clf_path)
        model.device = "cpu"
        model.to(model.device)
        return model

    def load_deeptrust_gpu():
        vect_path = "android-detectors/pretrained/deeptrust_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/deeptrust_classifier.pkl"
        model = DeepTrust.load(vect_path, clf_path)
        return model


    def load_rf():
        vect_path = "android-detectors/pretrained/random_forest_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/random_forest_classifier.pkl"
        return RF.load(vect_path, clf_path)


    def load_xgboost():
        vect_path = "android-detectors/pretrained/xgboost_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/xgboost_classifier.pkl"
        return XGBoost.load(vect_path, clf_path)


    def load_sadvnet():
        vect_path = "android-detectors/pretrained/guardnet_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/guardnet_classifier.pkl"
        model = GuardMLP.load(vect_path, clf_path)
        model.device = "cpu"
        model.to(model.device)
        return model


    def load_sadvnet_gpu():
        vect_path = "android-detectors/pretrained/guardnet_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/guardnet_classifier.pkl"
        model = GuardMLP.load(vect_path, clf_path)
        return model


    def load_wadvnet():
        vect_path = "android-detectors/pretrained/trustnet_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/trustnet_classifier.pkl"
        model = TrustMLP.load(vect_path, clf_path)
        model.device = "cpu"
        model.to(model.device)
        return model


    def load_wadvnet_gpu():
        vect_path = "android-detectors/pretrained/trustnet_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/trustnet_classifier.pkl"
        model = TrustMLP.load(vect_path, clf_path)
        return model


    def load_drebin():
        vect_path = "android-detectors/pretrained/drebin_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/drebin_classifier.pkl"
        return DREBIN.load(vect_path, clf_path)


    def load_secsvm():
        vect_path = "android-detectors/pretrained/secsvm_vectorizer.pkl"
        clf_path = "android-detectors/pretrained/secsvm_classifier.pkl"
        return SecSVM.load(vect_path, clf_path)

    model_registry = {
        # CPU Variants
        "DeepTrust": load_deeptrust,
        "RandomForest": load_rf,
        "XGBoost": load_xgboost,
        "SAdvNet": load_sadvnet,
        "wAdvNet": load_wadvnet,
        "DREBIN": load_drebin,
        "SecSVM": load_secsvm,
        # GPU Variants
        "DeepTrust_gpu": load_deeptrust_gpu,
        "SAdvNet_gpu": load_sadvnet_gpu,
        "wAdvNet_gpu": load_wadvnet_gpu,
    }

    target_models = []

    if args.model.lower() == "all":
        target_models = list(model_registry.keys())
    elif args.model in model_registry:
        target_models = [args.model]
    else:
        print(f"Error: Model '{args.model}' not found in registry.")
        print(f"Available models: {list(model_registry.keys())}")
        sys.exit(1)

    print(f"--- Running Evaluation for: {target_models} ---")

    for name in target_models:
        loader_func = model_registry[name]
        try:
            print(f"\n[Starting] {name}...")
            evaluate_inference_cost(loader_func, model_name=name)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            print(f"Skipping {name}.")