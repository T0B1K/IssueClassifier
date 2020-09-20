import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import joblib

import loadData
import LabelClassifier

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


labelClasses = ["enhancement", "bug", "doku", "api", ]
categories = [["enhancement", "bug"], ("doku", "api"), ["doku", "bug", "enhancement"]]#, ("doku", "bug"), ("api", "bug")]
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
    hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=True)
    for X_train, X_test, y_train, y_test in hue.getTrainingAndTestingData2():#labelClasses, categories):
        cat = categories[catIDX]
        print("\n--------- ( '{}', {} ) ---------".format(cat[0],str(cat[1:])))
        hue2 = LabelClassifier.LabelClassifier(cat)
        hue2.trainClassifier(X_train, y_train)
        prediction = hue2.predict(X_test)
        hue2.accuracy(X_test, y_test, prediction)

        #hue.createAntMapAndDocumentView(prediction, y_test, X_train, [cat])
        catIDX += 1

def predict(X_test):
    hue = loadData.DataPreprocessor(labelClasses, categories, loadVec=True)
    pretrained = joblib.load('../trainedClassifier/ensembleClassifier_doku-bug.joblib.pkl') 
    pretrained2 = joblib.load('../trainedClassifier/ensembleClassifier_doku-api.joblib.pkl')
    
    step = 400
    for i in range(0, len(X_test),step):
        trainigSilice = X_test[i:i+step]
        X = hue.Vecotrizer.transform(trainigSilice)
        classifier = LabelClassifier.LabelClassifier(("doku", "bug"), pretrained=pretrained)
        prediction = classifier.predict(X)
        labels = np.array(list(map ( lambda element : "doku" if element == 0 else "bug",prediction)))
    
        classifier2 = LabelClassifier.LabelClassifier(("doku", "api"), pretrained=pretrained2)
        prediction2 = classifier2.predict(X)
        labels2 = np.array(list(map ( lambda element : "doku" if element == 0 else "api",prediction2)))
        
        yield list(zip(trainigSilice, labels, labels2))


    #dummy für Aly
    #TODO lade alle classifier
    #Vergleiche bug, enhancement
    #Vergleiche restliche dinge vs doku
    #falls nicht doku, vergleiche restliche dinge vs api


initEverything()

tmp = predict(np.array(["bug, hilf mir", "hue, resolved doku"]))
for i in tmp:
    print(i)
