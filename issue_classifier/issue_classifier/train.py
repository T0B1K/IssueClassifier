from typing import Any, List

import joblib
import numpy as np

from issue_classifier.issue_data_methods.load_data import DataPreprocessor
from issue_classifier.issue_data_methods.load_data_antmap import antmap_preprocessor
from issue_classifier.label_classifier.label_classifier import LabelClassifier


def _init_classifier():
    catIDX = 0
    data_preprocessor = DataPreprocessor(
        label_classes, categories, load_vectorizer=True)

    for X_train, X_test, y_train, y_test in data_preprocessor.getTrainingAndTestingData2():
        category = categories[catIDX]
        print(
            "\n--------- ( '{}', {} ) ---------".format(category[0], str(category[1:])))
        label_classifier = LabelClassifier(category)
        label_classifier.trainClassifier(X_train, y_train)
        prediction = label_classifier.predict(X_test)
        label_classifier.accuracy(X_test, y_test, prediction)

        prediction_stacking = label_classifier.stackingPrediction(X_test)
        print("► ensemble-score:{}\n".format(np.mean(prediction_stacking == y_test)))
        #hue.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        catIDX += 1


def _load_pretrained_classifiers():
    pretrained_doku_bug_classifier = joblib.load(
        '../trainedClassifier/ensembleClassifier_doku-bug.joblib.pkl')
    pretrained_doku_api_classifier = joblib.load(
        '../trainedClassifier/ensembleClassifier_doku-api.joblib.pkl')
    return pretrained_doku_bug_classifier, pretrained_doku_api_classifier


def predict(X_test: List[str]) -> Any:
    data_preprocessor = DataPreprocessor(
        label_classes, categories, load_vectorizer=True)
    pretrained_doku_bug_classifier, pretrained_doku_api_classifier = _load_pretrained_classifiers()

    step = 400
    for i in range(0, len(X_test), step):
        trainigSilice = X_test[i:i+step]
        X = data_preprocessor.vectorzier.transform(trainigSilice)

        doku_bug_classifier = LabelClassifier(
            ("doku", "bug"), pretrained=pretrained_doku_bug_classifier)
        doku_bug_classifier_predictions = doku_bug_classifier.predict(X)
        doku_bug_classifier_labels = np.array(
            list(map(lambda element: "doku" if element == 0 else "bug", doku_bug_classifier_predictions)))

        doku_api_classifier = LabelClassifier(
            ("doku", "api"), pretrained=pretrained_doku_api_classifier)
        doku_api_classifier_predictions = doku_api_classifier.predict(X)
        doku_api_classifier_labels = np.array(
            list(map(lambda element: "doku" if element == 0 else "api", doku_api_classifier_predictions)))

        yield list(zip(trainigSilice, doku_bug_classifier_labels, doku_api_classifier_labels))


label_classes = ["enhancement", "bug"]
categories = [("enhancement", "bug")]

training_percentage = 0.7

antmap = antmap_preprocessor(label_classes, categories)
category_index = 0

for X_train, X_test, y_train, y_test in antmap.get_training_and_testing_data(label_classes, categories):
    category = categories[category_index]

    label_classifier = LabelClassifier(category)
    label_classifier.trainClassifier(X_train, y_train)

    prediction = label_classifier.predict(X_test)

    antmap.create_antmap_and_document_view(
        prediction, y_test, X_train, [category])
    print("► ensemble-score:{}\n".format(np.mean(prediction == y_test)))

    category_index += 1
