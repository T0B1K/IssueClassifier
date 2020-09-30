import loadDataAntmap
import fileManipulation
import loadData
import LabelClassifier
import joblib
import numpy as np

labelClasses = ["enhancement", "bug", "doku", "api"]
categories = [("doku", "bug"), ("doku", "api")]#, ("doku", "api"), ["doku", "bug", "enhancement"]]#, ("doku", "bug"), ("api", "bug")]
trainingPercentage = fileManipulation.FileManipulation.values["trainingPercentage"]  # This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing



#Pascal load aus einem Dokument
classifiers = [joblib.load('../trainedClassifier/ensembleClassifier_doku-api.joblib.pkl'),
    joblib.load('../trainedClassifier/ensembleClassifier_doku-bug.joblib.pkl')]
classifierClassification = [("bug", "enhancement"), ("doku", "nodoku")]

classifiers = [i for i in zip(classifierClassification, classifiers)]

classifiers = list(map(lambda labels, pretrained: (labels,LabelClassifier.LabelClassifier(labels, pretrained=pretrained)), classifiers))

hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=True)


def predict(X_test):
    step = 400
    for i in range(0, len(X_test),step):
        X = vectorizerMiddleWare(X_test[i:i+step])
        yield classificationUsingTree(X)

def vectorizerMiddleWare(X_test):
    return hue.Vecotrizer.transform(X_test)

def classificationUsingTree(X_test):
    #todo Tree logic
    returnLabels = [X_test]
    for labelPrediction, classifier in classifiers:
        prediction = classifier.predict(X_test)
        labels = np.array(list(map ( lambda element : labelPrediction[0] if element == 0 else labelPrediction[1],prediction)))

        returnLabels = list(zip(returnLabels, labels))  #change logic for tree
    return returnLabels

    #dummy für Aly
    #TODO lade alle classifier
    #Vergleiche bug, enhancement
    #Vergleiche restliche dinge vs doku
    #falls nicht doku, vergleiche restliche dinge vs api

#tmp = predict(np.array(["bug, hilf mir", "hue, resolved doku"]))
#for i in tmp:
#    print(i)


