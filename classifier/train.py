import matplotlib.pyplot as plt
import numpy as np
import nltk

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

from nltk.stem import WordNetLemmatizer, PorterStemmer
#nltk.download('wordnet')

import seaborn as sn
import pandas as pd

import loadData
import joblib

labelClasses = ["bug", "enhancement"]#, "api", "doku"]
categories = [("bug", "enhancement")]#, ("api", "bug"), ("doku", "bug")]
estimators=[('MultinomialNB', MultinomialNB()), \
    ('SGDClassifier', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)),
    ('sigmoidSVM', SVC(kernel='sigmoid', gamma=1.0)),
    ('RandomForest', RandomForestClassifier(200, bootstrap=False)),
    #('BernoulliNB', BernoulliNB()),#the worst one
    ('LogisticRegression',LogisticRegression(solver='sag',random_state=100))
    ]

trainingPercentage=0.7  #This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing
listOfPropab = []

#This method is used to train the given classifiers
def trainClassifiers(X_train_featureVector, X_test_featureVector, y_train, y_test, classifierObjectList, cat = ("bug", "feature")):
    global listOfPropab
    predictions = []
    for classifierName,classifier in classifierObjectList:
        print(classifierName)
        classifier.fit(X_train_featureVector, y_train)
        predicted = classifier.predict(X_test_featureVector)
        scores.append(np.mean(predicted == y_test))
        print(metrics.classification_report(y_test, predicted,labels=[0,1], target_names=[cat[0],cat[1]]))
        if not classifierName == "sigmoidSVM":
            listOfPropab.append(classifier.predict_proba(X_test))
        predictions.append(predicted)
    return predictions
    

#---------------------
hue = loadData.DataPreprocessor()

folder = '../trainedClassifier/'
newClassifier = True
catIDX = 0

for X_train, X_test, y_train, y_test in hue.getTrainingAndTestingData(labelClasses, categories):
    print("training on: {}% == {} documents\ntesting on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))
    #classifierPredictions = trainClassifiers(X_train, X_test, y_train, y_test, estimators, categories[catIDX])
    
    #TODO speichere den TFIDF Vektorizer, da es sonst zu einem dimmension missmatch kommt
    ensemble = None
    nameAddon = "_{}-{}.joblib.pkl".format(categories[catIDX][0],categories[catIDX][1])
    tmpName = folder + "ensembleClassifier" + nameAddon
    if newClassifier:
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(X_train, y_train)#test our model on the test data
        _ = joblib.dump(ensemble, tmpName, compress=9)
        plot_confusion_matrix(ensemble, X_test, y_test, normalize="all",display_labels=[categories[catIDX][0],categories[catIDX][1]])
    else:
        ensemble = joblib.load(tmpName)
    
    finalPrediction = ensemble.predict(X_test)

    print("ensemble-score:{}".format(np.mean(finalPrediction == y_test)))
    print("trained on: {}% == {} documents\ntested on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))
    plt.show()

    hue.createAntMapAndDocumentView(finalPrediction, y_test, X_train, [categories[catIDX]])
    catIDX += 1