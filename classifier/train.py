import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import joblib

import loadData
import LabelClassifier

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


labelClasses = ["enhancement", "bug"]#, "doku", "api", ]
categories = [("enhancement", "bug")]#, ("doku", "bug"), ("api", "bug")]
trainingPercentage = 0.7  # This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing

"""
                                 ╔══ ► Classifier 1  ═╗
                                 ║                    ║
[Queue] Daten ► Vectorizer  ═════╬══ ► Classifier 2  ═╬═ ► TODO Prediction bassierend auf den anderen [array prediction] ► [Queue]
             [Vorverarbeitung]   ║                    ║
                                 ╚══ ►    ...        ═╝
[Nach der Vorverarbeitung können wir *später* eine Exchange Queue bauen]
TODO: Queue Daten [überschreiben] das X_test, rufen die Funktion .predct(X_test) eines (pretrained) klassifiers auf
        das von der Methode returnte wird wieder in eine queue geschrieben

WORK in progress: Neue predict funktion, die automatisch alle labels bei predicted zurückgibt (also predict für eine TODO Überklasse)
"""

#pretrained = None #joblib.load('../trainedClassifier/ensembleClassifier_doku-api.joblib.pkl') 

#hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=False)
#hue2 = LabelClassifier.LabelClassifier(("doku", "api"), pretrained=pretrained)

#X_train, X_test, y_train, y_test = next(hue.getTrainingAndTestingData(labelClasses, categories))

#hue2.trainClassifier(X_train, y_train)
#hue2.accuracy(X_test, y_test)
#prediction = hue2.predict(X_test)

def initEverything():
    catIDX = 0
    hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=False)
    for X_train, X_test, y_train, y_test in hue.getTrainingAndTestingData(labelClasses, categories):
        cat = categories[catIDX]
        print("\n--------- ( {}, {} ) ---------".format(cat[0],cat[1]))
        hue2 = LabelClassifier.LabelClassifier(cat)
        hue2.trainClassifier(X_train, y_train)
        prediction = hue2.predict(X_test)
        hue2.accuracy(X_test, y_test, prediction)

        hue.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        catIDX += 1

def predict(X_test):
    #dummy für Aly
    #TODO lade alle classifier
    #Vergleiche bug, enhancement
    #Vergleiche restliche dinge vs doku
    #falls nicht doku, vergleiche restliche dinge vs api
    return None

initEverything()