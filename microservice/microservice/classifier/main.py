import os

import joblib
import numpy as np
import classification_tree
import configuration

FOLDER_PATH = os.environ["CLASSIFIER_PATH"] or "./classifier/trained_classifiers/"

trained_classifiers = [joblib.load(FOLDER_PATH + 'ensembleClassifier_enhancement-bug.joblib.pkl'),
                     joblib.load(FOLDER_PATH + 'ensembleClassifier_doku-api.joblib.pkl')]
classifier_categories = [("enhancement", "bug"), ("doku", "api")]
config  = configuration.Configuration()
tree = classification_tree.ClassificationTree(config.getValueFromConfig("labelClasses"))


def classify_issues(vectorised_issues) -> np.array:
    """
        This method predicts the output given by a vectorrizer output and returns the labels zipped as list

        Input-representation of the numpy array: [[1,2,7,0,4], [0,2,67,3,1], ...]

        returns: [["bug", "api"], ["enhancement"], ...]
    """
    prediction = tree.classify(vectorised_issues)
    # prediction = np.array(list(map(
    #     lambda x: classifier_categories[0][0] if x == 0 else classifier_categories[0][1], prediction)))
    return prediction
