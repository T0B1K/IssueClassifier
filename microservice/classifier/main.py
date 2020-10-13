import joblib
import numpy as np

FOLDER_PATH = "./classifier/trainedClassifiers/"

trained_classifiers = [joblib.load(FOLDER_PATH + 'ensembleClassifier_enhancement-bug.joblib.pkl'),
                       joblib.load(FOLDER_PATH + 'ensembleClassifier_doku-api.joblib.pkl')]
classifier_categories = [("enhancement", "bug"), ("doku", "api")]


def predict(vectorrizerOutput) -> np.array:
    """
        This method predicts the output given by a vectorrizer output and returns the labels zipped as list
        Input-representation of the numpy array: [[1,2,7,0,4], [0,2,67,3,1], ...]
        returns: [["bug", "api"], ["enhancement"], ...]
    """

    final_labels = []*vectorrizerOutput.shape[0]
    for classifier, label in zip(trained_classifiers, classifier_categories):
        prediction = classifier.predict(vectorrizerOutput)
        prediction = np.array(
            list(map(lambda x: label[0] if x == 0 else label[1], prediction)))
        final_labels.push(prediction)
    return list(zip(final_labels))
