# DeepTrust's Official Repository

This is the official repository of DeepTrust, an Android Malware Detection System winner of the
[Robust Android Malware Detection Competition at 2025 IEEE Conference SaTML](https://ramd-competition.github.io/).

This project is a fork of the original repository of the competition, which can be found at
[elsa-cybersecurity](https://github.com/pralab/elsa-cybersecurity?tab=readme-ov-file#tldr-how-to-participate). It adds necessary code and dependencies for the DeepTrust submission to the competition and experiments
carried out in the research paper. Read the inherited [elsa-cyber_README](elsa-cyber_README.md) file of the repository
for more information on the benchmark.

### Research
Xxx

### Folder Structure
    deetrust
    ├── README.md # This file
    ├── elsa-cyber_README.md # General README of repositoy competition
    ├── android-detectors # Baselines and required code for model implementation
    ├── config.py # Configuration file for the whole repository
    ├── data # Place the datasets here
    ├── experiments # Code for the experiments in the research paper
    │   ├── configs
    │   ├── evaluation.py
    │   ├── out
    │   └── training.py
    ├── main.py # Main script to run the different tracks of the competition
    ├── notebooks # Jupyter notebooks for research figures
    │   ├── embeddings
    │   ├── figures
    │   └── nb-research-figures.ipynb
    ├── submissions # Place the submission files here
    ├── track_1
    │   ├── attack_requirements.txt
    │   ├── deeptrust_requirements.txt
    │   ├── evaluation.py
    │   ├── feature_space_attack
    │   ├── iiia_config # Own configuration files for hyperparameter search
    │   ├── iiia_evaluate.py # Own script to evaluate a trained model
    │   └── iiia_hyperparameter_bo.py # Own script for hyperparameter search
    ├── track_2
    │   ├── apk_downloader.py
    │   ├── attack_requirements.txt
    │   ├── evaluation.py
    │   └── problem_space_attack
    └── track_3
        ├── apk_downloader.py
        └── evaluation.py

## Reproduce DeepTrust's results in the Benchmark

Follow the instructions to produce the submission files for the competition.

1. Download the training dataset, the Track 1, Track 2 and Track 3 test datasets and their pre-extracted features from the [ELSA benchmarks website](https://benchmarks.elsa-ai.eu/?ch=6&com=downloads).
2. Place the datasets without uncompressing them inside the `data` directory.
2. It is recommended to create a new environment. We use conda (it might be required to append `android-detectors/src` directory to the python path before launching the script).
    ```bash
   # Create a new conda environment and install the required dependencies
    conda create -n android python=3.9
    conda activate android
    pip install -r android-detectors/requirements.txt
    pip install -r track_1/attack_requirements.txt
    pip install -r track_1/deeptrust_requirements.txt
    pip install -r track_2/attack_requirements.txt
    pip install -r track_2/problem_space_attack/manipulation/Obfuscapk/src/requirements.txt
   
   # Set PATH variables
    export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
   ## Only for Track 2
    export PATH=$PATH:~/android-sdk/build-tools/34.0.0
    ```

DeepTrust is a multi-step system that is compound of three learners, two MLPs and an Isolation Forest placed in the second activation condition: SAdvNet (_GuardNet_), wAdvNet (_TrustNet_), Isolation Forest (_InspectRF_).
Therefore, you can reproduce the results from different starting points:
1. Using DeepTrust trained.
2. Using SAdvNet and wAdvNet trained and then train the Isolation Forest within DeepTrust.
3. Training all the components from scratch.

Please, notice that reproducing the results may take a while, so we recommend to run each Track in different sessions

### 1. Reproduce using DeepTrust trained.

The files required for it are in:
https://drive.google.com/drive/folders/1MzppCM60UBRjTAZ5jBm32Pfo0if21YmX?usp=sharing. Download them and place them in the `android-detectors/pretrained` directory. 
These are:
- `deeptrust_classifier.pkl`: InspectRF pickle file
- `deeptrust_classifier.pth`: TrustNet and GuardNet PyTorch weights
- `deeptrust_vectorizer.pkl`: DeepTrust vectorizer (for feature extraction)
- `guardnet_vectorizer.pkl`: GuardNet vectorizer (for feature extraction)
- `trustnet_vectorizer.pkl`: TrustNet vectorizer (for feature extraction)
```bash
# Run the main script with the pretrained model

## Track 1: Robustness to Feature-space Attack
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust

## Track 2: Robustness to Problem-space Attack
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust

## Track 3: Robustness to Data Drift
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 3 --method_name deeptrust
```

### 2. Reproduce using SAdvNet and wAdvNet trained and then train the Isolation Forest within DeepTrust.
The files required for it are in:
https://drive.google.com/drive/folders/1MzppCM60UBRjTAZ5jBm32Pfo0if21YmX?usp=sharing. Download them and place them in the `pretrained` directory. 
These are:
- `guardnet_classifier.pkl`: GuardNet PyTorch weights
- `guardnet_vectorizer.pkl`: GuardNet vectorizer (for feature extraction)
- `trustnet_classifier.pkl`: TrustNet PyTorch weights
- `trustnet_vectorizer.pkl`: TrustNet vectorizer (for feature extraction)

Run the same commands as in 1. The code handles the training of the Isolation Forest within DeepTrust in case
it is missing.

Once DeepTrust is trained, you can delete the `trustnet_classifier.pkl` and `guardnet_classifier.pkl` 
files from the `pretrained` directory as their weights are already saved within the `deeptrust_classifier.pth` file.

### 3. Reproduce training all the components from scratch.
Please, notice that this may take a while, so we recommend to run each Track in different sessions.

```bash
# Run the main script to train the classifiers from scratch and produce the submission files
## Train SAdvNet and evaluate on Track 1
python main.py --clf_loader_path android-detectors/src/loaders/trust_mlp_loader.py --track 1 --method_name trust_mlp

## Train wAdvNet and evaluate on Track 1
python main.py --clf_loader_path android-detectors/src/loaders/guard_mlp_loader.py --track 1 --method_name guard_mlp

## Train Isolation Forest and evaluate DeepTrust on Track 1
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust

## Track 2
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust

## Track 3
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 3 --method_name deeptrust
```

### Note: On Track 2 Submission.

The Track 2 submission requires the use of Obfuscapk to obfuscate the APKs.
We recommend to follow the instructions on the [Obfuscapk repository](track_2/problem_space_attack/manipulation/Obfuscapk/README.md) to install it. 

In this case, this repository already have
Obfuscapk from source (if somehow is missing, download it and place it in `track_2/problem_space_attack/manipulation/Obfuscapk`), 
so you would only need to have a recent version of
[`apktool`](https://ibotpeaches.github.io/Apktool/),
[`apksigner`](https://developer.android.com/studio/command-line/apksigner)
and [`zipalign`](https://developer.android.com/studio/command-line/zipalign) installed
and available from the command line, as indicated in the instructions of the [Obfuscapk repository](track_2/problem_space_attack/manipulation/Obfuscapk/README.md).

For that:

1. Install the command line tools of the Android SDK, which can be downloaded from the 
[Android developer website](https://developer.android.com/studio#command-line-tools-only). 
2. Install the build tools version 34.0.0, which is the one used in this repository (other versions could be incompatible). 
3. Place it in the `~/android-sdk/build-tools/34.0.0` directory, 
or place it somewhere else and change the path in the `export PATH` command below.

Furthermore, Track 2 works at APK level, so you will need to place an API Key into the `config.py` so the APKs can be 
downloaded from [AndroZoo](https://androzoo.uni.lu/) for the first time you run the command below. Note that this is a long process that
takes up considerable disk space. The attack of Track 2 does not work on _MacOS_ (acknowledged by the organizers), so it is recommended to run it on a Linux platform.

```bash
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
# Change by the path where you have installed the Android SDK build tools
export PATH=$PATH:~/android-sdk/build-tools/34.0.0
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust
```

## Reproduce Research Paper Experiments
The experiments conducted in _Section 5.2, Evaluation > Ablation study_ are implemented in the `experiments` directory.
1. Training:
   - In the `experiments/training.py` file, you can find the code. There are blocks of code to train
   each of the _experiment groups_. If you want to run a specific experiment, you can comment out the others.
   and then run the script
    ```bash
   export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
   python experiments/training.py
    ```
   - The trained models will be saved in different directories within the `experiments/out`. In each directory you will
   find four files: 
     - `config.yaml`: The configuration setup of the model
     - `vectorizer.pkl`: The vectorizer used for the model.
     - `classifier.ckpt`: The weights of the model.
     - `train_metrics.json`: The training metrics of the model.
2. Evaluation:
    - In the `experiments/evaluation.py` file, you can find the code to evaluate the models trained in the previous step.
    - You can run the script with the following command:
   ```bash
    export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
    python experiments/evaluation.py --models_dir experiments/out
    ```
    - If the `--models_dir` argument points to a single model directory (containing `config.yaml`, `vectorizer.pkl`, and `classifier.ckpt`), the script will evaluate that specific model.
    - If the `--models_dir` argument points to a directory containing multiple model directories, the script will automatically detect and evaluate all models in parallel.
    - The evaluation results will be saved in each model's directory as `evaluation.json`, which contains the evaluation metrics for the model in Track 1 of the benchmark.

In this repository we provide the following files for all the experiments:
- `config.yaml`: The configuration setup of the model.
- `train_metrics.json`: The training metrics of the model.
- `evaluation.json`: The test metrics of the model.

We do not provide the trained models, as they are too large to be uploaded to GitHub. You can either train them yourself using the `experiments/training.py` script 
or download them from the following link: https://drive.google.com/drive/folders/13mVX38l-Ibk7wE59zaVTHObBymtQAk24?usp=drive_link