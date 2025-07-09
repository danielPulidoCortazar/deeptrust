import os
import models
from models import RobustMLP
from models.utils import *


def load():
    """
    NB: in this example, the pre-extracted features are used. Alternatively,
    the APK file paths can be passed to the classifier.
    To fit the model, you can use `classifier.extract_features` to get the
    features and then pass them to `classifier.fit`.
    To classify the APK files, you can directly pass the list containing the
    file paths to `classifier.classify`.
    """

    project_root_path = os.path.join(os.path.dirname(models.__file__), "../../..")

    clf_path = os.path.normpath(os.path.join(
        project_root_path, "android-detectors/pretrained/robust_mlp_classifier.ckpt"))
    vect_path = os.path.normpath(os.path.join(
        project_root_path, "android-detectors/pretrained/robust_mlp_vectorizer.pkl"))
    config_path = os.path.normpath(os.path.join(
        project_root_path, "experiments/configs/robust_mlp.yaml"))

    if os.path.exists(clf_path) and os.path.exists(vect_path):
        print("Pre-trained model found. Loading model")
        classifier = RobustMLP.load(config_path, vect_path, clf_path)
    else:
        print("Pre-trained model not found. Training new model")
        classifier = RobustMLP(config_path)
        features_tr = load_features("data/training_set_features.zip")
        y_tr = load_labels("data/training_set_features.zip",
                           "data/training_set.zip")
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    return classifier