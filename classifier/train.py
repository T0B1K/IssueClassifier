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

import json 
import seaborn as sn
import pandas as pd

import loadData

labelClasses = ["bug", "enhancement", "api", "doku"]
categories = [("bug", "enhancement"), ("api", "bug"), ("doku", "bug")]
estimators=[('MultinomialNB', MultinomialNB()), \
    ('SGDClassifier', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)),
    ('sigmoidSVM', SVC(kernel='sigmoid', gamma=1.0)),
    ('RandomForest', RandomForestClassifier(200, bootstrap=False)),
    #('BernoulliNB', BernoulliNB()),#the worst one
    ('LogisticRegression',LogisticRegression(solver='sag',random_state=100))
    ]

scores = []             #scores of the classifiers
X_test_documents = None #the texts of the test documents
documentTexts = None    #all the documents
trainingPercentage=0.7  #This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing
permutation = None      #how the data was permuted

numberToWordMapping = None
tidvectorizer = None
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
    

#This method is used for ensemble learning, to get a majority decission based on the weights
#it returns the decission of the classifiers
def classifierVoting(classifiers, x_train, x_test, y_train ,voteStrength=None, cat = ("bug", "feature")):
    ensemble = VotingClassifier(estimators,weights=voteStrength, voting='hard')
    #fit the esemble model, after fitting the other ones
    ensemble.fit(x_train, y_train)#test our model on the test data
    plot_confusion_matrix(ensemble, x_test, y_test, normalize="all",display_labels=[cat[0],cat[1]])
    return ensemble.predict(x_test)

#---------------------
hue = loadData.DataPreprocessor()

catIDX = 0
for X_train, X_test, y_train, y_test in hue.getTrainingAndTestingData(labelClasses, categories):
    print("training on: {}% == {} documents\ntesting on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))
    #classifierPredictions = trainClassifiers(X_train, X_test, y_train, y_test, estimators, categories[catIDX])

    finalPrediction = classifierVoting(estimators, X_train, X_test, y_train)
    scores = []

    print("ensemble-score:{}".format(np.mean(finalPrediction == y_test)))
    print("trained on: {}% == {} documents\ntested on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))
    plt.show()
    #saveWrongPredictions(y_test, finalPrediction, "wrongClassified.json")

    hue.createAntMapAndDocumentView(finalPrediction, y_test, X_train, [categories[catIDX]])
    catIDX += 1