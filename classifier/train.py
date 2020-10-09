import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import joblib

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import loadDataAntmap
import fileManipulation
import loadData
import LabelClassifier

labelClasses = ["enhancement", "bug", "doku", "api"]
categories = [("doku", "bug"), ("doku", "api")]#, ("doku", "api"), ["doku", "bug", "enhancement"]]#, ("doku", "bug"), ("api", "bug")]
trainingPercentage = fileManipulation.FileManipulation.values["trainingPercentage"]  # This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing

#pretrained = None #joblib.load('../trainedClassifier/ensembleClassifier_doku-api.joblib.pkl') 

#hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=False)
#hue2 = LabelClassifier.LabelClassifier(("doku", "api"), pretrained=pretrained)

#X_train, X_test, y_train, y_test = next(hue.getTrainingAndTestingData(labelClasses, categories))

#hue2.trainClassifier(X_train, y_train)
#hue2.accuracy(X_test, y_test)
#prediction = hue2.predict(X_test)

"""
amp = loadDataAntmap.AntMapPreprozessor(labelClasses, categories)
catIDX = 0

for X_train, X_test, y_train, y_test in amp.getTrainingAndTestingData(labelClasses, categories):
    cat = categories[catIDX]
    hue2 = LabelClassifier.LabelClassifier(cat)
    hue2.trainClassifier(X_train, y_train)
    prediction = hue2.predict(X_test)
    amp.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
    print("► ensemble-score:{}\n".format(np.mean(prediction == y_test)))

    catIDX += 1
"""

def initEverything():
    catIDX = 0
    hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=True)
    for X_train, X_test, y_train, y_test in hue.getTrainingAndTestingData2():#labelClasses, categories):
        cat = categories[catIDX]
        print("\n--------- ( '{}', {} ) ---------".format(cat[0],str(cat[1:])))
        hue2 = LabelClassifier.LabelClassifier(cat)
        hue2.trainClassifier(X_train, y_train, loadClassifier=False)
        prediction = hue2.predict(X_test)
        hue2.accuracy(X_test, y_test, prediction)

        prediction2 = hue2.stackingPrediction(X_test)
        print("► ensemble-score:{}\n".format(np.mean(prediction2 == y_test)))
        #hue.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        catIDX += 1


initEverything()

