from typing import List

import joblib
from microservice.config.classifier_config import Configuration

config = Configuration()
classifier_locations = config.get_value_from_config("classifier classifierLocations")
root_folder = config.get_value_from_config("classifier path loadFolder")


def get_classifier(labels: List[str]):
    if not labels:
        raise Exception("There are no categories provided")

    classifier_path = None
    for classifier_location in classifier_locations:
        if classifier_location["labels"] == labels:
            classifier_path = classifier_location["path"]
    _path: str = "{}/{}".format(root_folder, classifier_path)
    assert classifier_path is not None, "Labels: {}".format(labels)

    classifier = joblib.load(_path)
    assert classifier is not None, "Classifier couldn't be loaded from {}".format(_path)

    return classifier


def get_voting_classifier():
    classifier_path = config.get_value_from_config("trainingConstants voting")
    path: str = "{}/{}".format(root_folder, classifier_path)
    classifier = joblib.load(path)

    assert classifier is not None, "Classifier at {} couldn't be loaded".format(path)

    return classifier


def get_vectoriser():
    _vectoriser_path = config.get_value_from_config("vectorizer path loadPath")
    vectoriser = joblib.load(_vectoriser_path)

    assert vectoriser is not None, "Vectoriser at {} couldn't be loaded".format(
        _vectoriser_path
    )

    return vectoriser
