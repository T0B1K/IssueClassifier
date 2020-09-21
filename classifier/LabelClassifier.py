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
from sklearn.kernel_approximation import RBFSampler

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
        self.stackingEstimator = None
        self.rbfKernel = None
    
    def trainClassifier(self, X_train, y_train,loadClassifier = True, saveToFile = True):
        print("> training classifier")
        voting = None
        if loadClassifier == True:
            try:
                self.trainedEstimator = joblib.load(self.fileLocation)
                voting = joblib.load('../classifier/VotingClassifier')
            except:
                print("Classifier could not be loaded")
                raise
                
        else:
            self.trainedEstimator = VotingClassifier(self.estimators, voting='hard')
            voting = self.trainedEstimator.fit_transform(X_train, y_train) # test our model on the test data
            if saveToFile == True:
                joblib.dump(self.trainedEstimator , self.fileLocation, compress=9)
                joblib.dump(voting, '../classifier/VotingClassifier', compress=9)
            print("> dumped Classifier: {}".format(self.fileLocation))
        self.trainKernelApproxSvgOnVoting(voting, y_train)

    def predict(self, X_test):
        print("> predicting")
        return self.trainedEstimator.predict(X_test)

    def generateFilename(self, folder = '../trainedClassifier/'):
        return "{}ensembleClassifier_{}-{}.joblib.pkl".format(folder, self.category[0], self.category[1])

    def accuracy(self, X_test, y_test, predicted):
        if self.trainedEstimator == None:
            raise AssertionError("Classifier has not been trained yet")
        print("\n► ensemble-score:{}\n".format(np.mean(predicted == y_test)))
        plot_confusion_matrix(self.trainedEstimator, X_test, y_test, normalize="all",display_labels=[self.category[0],self.category[1]])
        plt.show()
    
    def trainKernelApproxSvgOnVoting(self, X_predicted, y):
        print("training stacking classifier")
        self.rbfKernel = RBFSampler(gamma=1, random_state=1)
        X_features = self.rbfKernel.fit_transform(X_predicted)
        self.stackingEstimator = SGDClassifier(max_iter=1000)
        self.stackingEstimator.fit(X_features, y)
        print("stacking-classifier: " + str(self.stackingEstimator.score(X_features, y)))
    
    def stackingPrediction(self, X_test):
        voting = self.trainedEstimator.transform(X_test)
        influencedVoting = self.rbfKernel.transform(voting)
        return self.stackingEstimator.predict(influencedVoting)
