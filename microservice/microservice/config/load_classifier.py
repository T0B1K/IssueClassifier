import joblib
from microservice.config.classifier_configuration import Configuration

config = Configuration()

classifier_locations = config.get_value_from_config("classifierLocations")
root_folder = config.get_value_from_config("classifierFolder")


def get_classifier(categories: list):
    # if not categories:
    #     raise Exception("There are no categories provided")

    classifier_path = None
    for element in classifier_locations:
        if element["labels"] == categories:
            classifier_path = element["path"]
    path: str = "{}/{}".format(root_folder, classifier_path)
    # assert classifier_path is not None, "Categories: {}".format(categories)
    classifier = joblib.load(path)

    # assert classifier is not None, "Classifier couldn't be loaded from {}".format(path)

    return classifier


def get_voting_classifier():
    classifier_path = config.get_value_from_config("trainingConstants voting")
    path: str = "{}/{}".format(root_folder, classifier_path)
    classifier = joblib.load(path)

    # assert classifier is not None, "Classifier at {} couldn't be loaded".format(path)

    return classifier


def get_vectoriser():
    vectoriser_path = config.get_value_from_config("vectorizer path loadPath")
    vectoriser = joblib.load(vectoriser_path)

    # assert vectoriser is not None, "Vectorizer at {} couldn't be loaded".format(
    #     vectoriser_path
    # )

    return vectoriser
