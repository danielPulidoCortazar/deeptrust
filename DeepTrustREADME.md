## Reproduce DeepTrust results in the Benchmark

_First, read the general [README](README.md) file of the repository._

This project is a fork of the original repository of the competition, which can be found at
[elsa-cybersecurity](https://github.com/pralab/elsa-cybersecurity?tab=readme-ov-file#tldr-how-to-participate).
It only add necessary code and dependencies for the DeepTrust submission and experiments.

These commands can be used to produce the submission files for DeepTrust classifier.

Download the training dataset, the Track 1, Track 2 and Track 3 datasets and their pre-extracted features from the [ELSA benchmarks website](https://benchmarks.elsa-ai.eu/?ch=6&com=downloads) inside the `data` directory.

It is recommended to create a new environment. In this example we use conda (it might be required to append `android-detectors/src` directory to the python path before launching the script).

### Reproduce results with trained model
Please, notice that this may take a while, so we recommend to run each Track in a different sessions

The files required for it are in:
https://drive.google.com/drive/folders/1MzppCM60UBRjTAZ5jBm32Pfo0if21YmX?usp=sharing. Download them and place them in the `pretrained` directory. 
These are:
- `deeptrust_classifier.pkl`: InspectRF pickle file
- `deeptrust_classifier.pth`: TrustNet and GuardNet weights
- `deeptrust_vectorizer.pkl`: DeepTrust vectorizer
- `guardnet_vectorizer.pkl`: GuardNet vectorizer
- `trustnet_vectorizer.pkl`: TrustNet vectorizer
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
export PATH=$PATH:~/android-sdk/build-tools/34.0.0 # For Track 2

# Run the main script with the pretrained model
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 3 --method_name deeptrust
```

### Reproduce results training from scratch
Please, notice that this may take a while, so we recommend to run each Track in a different sessions

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
export PATH=$PATH:~/android-sdk/build-tools/34.0.0 # For Track 2

# Run the main script to train the classifiers from scratch and produce the submission files
python main.py --clf_loader_path android-detectors/src/loaders/trust_mlp_loader.py --track 1 --method_name trust_mlp
python main.py --clf_loader_path android-detectors/src/loaders/guard_mlp_loader.py --track 1 --method_name guard_mlp
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 1 --method_name deeptrust
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 3 --method_name deeptrust
```

Once DeepTrust is trained, you can delete the `trustnet_classifier.pkl` and `guardnet_classifier.pkl` 
files from the `pretrained` directory as their weights are already saved in the `deeptrust_classifier.pth` file.

**On Track 2 Submission**

The Track 2 submission is a bit different, as it requires the use of Obfuscapk to obfuscate the APKs.
We recomment to follow the instructions in the [Obfuscapk repository](track_2/problem_space_attack/manipulation/Obfuscapk/README.md) to install it. 

In this case, this repository already have
Obfuscapk from source, so you would only need to have a recent version of
[`apktool`](https://ibotpeaches.github.io/Apktool/),
[`apksigner`](https://developer.android.com/studio/command-line/apksigner)
and [`zipalign`](https://developer.android.com/studio/command-line/zipalign) installed
and available from the command line, as indicated in the instructions of the [Obfuscapk repository](track_2/problem_space_attack/manipulation/Obfuscapk/README.md).

It is enough to install the command line tools of the Android SDK, which can be downloaded from the 
[Android developer website](https://developer.android.com/studio#command-line-tools-only). And then install the build tools version 34.0.0, 
which is the one used in this repository.

Furthermore, you will need to provide an Api key into the `config.py` so the APKs can be downloaded from the AndroZoo repository.

Once everything is set up, you can run the following command to produce the submission files for Track 2. In order to reproduce the submission
all the APKs will be downloaded, which is a long process. Also the evasion attack in this track is quite slow and does not work on MacOS, 
so it is recommended to run it on a Linux machine.
```bash
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
export PATH=$PATH:~/android-sdk/build-tools/34.0.0 # Add the path for access to the Android SDK build tools
python main.py --clf_loader_path android-detectors/src/loaders/deeptrust_loader.py --track 2 --method_name deeptrust
```

## Reproduce Research Paper Experiments
The experiments conducted in section 4.2 of the research paper are implemented in the `experiments` directory.
1. Training:
   - In the `experiments/training.py` file, you can find the code. There, there are blocks of code to train
   each of the _experiment groups_. If you want to run a specific experiment, you can comment out the others.
   and then run the script
    ```bash
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
    python experiments/evaluation.py --models_dir experiments/out
    ```
    - If the `--models_dir` argument points to a single model directory (containing `config.yaml`, `vectorizer.pkl`, and `classifier.ckpt`), the script will evaluate that specific model.
    - If the `--models_dir` argument points to a directory containing multiple model directories, the script will automatically detect and evaluate all models in parallel.
    - The evaluation results will be saved in each model's directory as `evaluation.json`, which contains the evaluation metrics for the model in Track 1 of the benchmark.

In this repository we provide the following files for all the experiments. We do not provide the trained models, 
as they are too large to be uploaded to GitHub.
- `config.yaml`: The configuration setup of the model.
- `train_metrics.json`: The training metrics of the model.
- `evaluation.json`: The test metrics of the model.