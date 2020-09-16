import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

class LabelClassifier:
    def __init__(self, categoryToClassify, pretrained = None, folder2Save = '../trainedClassifier/'):
        self.category = categoryToClassify
        self.estimators = estimators=[('MultinomialNB', MultinomialNB()), \
        ('SGDClassifier', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)),
        ('sigmoidSVM', SVC(kernel='sigmoid', gamma=1.0)),
        ('RandomForest', RandomForestClassifier(200, bootstrap=False)),
        ('LogisticRegression',LogisticRegression(solver='sag',random_state=100))]
        self.trainedEstimator = pretrained
        self.fileLocation = self.generateFilename(folder2Save)
    
    def trainClassifier(self, X_train, y_train, saveToFile = True):
        print("> triaining classifier")
        self.trainedEstimator = VotingClassifier(self.estimators, voting='hard')
        self.trainedEstimator.fit(X_train, y_train) # test our model on the test data
        joblib.dump(self.trainedEstimator , self.fileLocation, compress=9)
        print("> dumped Classifier: {}".format(self.fileLocation))

    def predict(self, X_test):
        print("> predicting")
        return self.trainedEstimator.predict(X_test)

    def generateFilename(self, folder = '../trainedClassifier/'):
        return "{}ensembleClassifier_{}-{}.joblib.pkl".format(folder, self.category[0], self.category[1])

    def accuracy(self, X_test, y_test, predicted):
        if self.trainedEstimator == None:
            raise AssertionError("Classifier has not been trained yet")
        print("\n» ensemble-score:{}\n".format(np.mean(predicted == y_test)))
        plot_confusion_matrix(self.trainedEstimator, X_test, y_test, normalize="all",display_labels=[self.category[0],self.category[1]])
        plt.show()
        