import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import loadClassifier

import logging

class LabelClassifier:
    """Class implemens various label Classifiers """

    def __init__(self, categoryToClassify, pretrained = None, folder2Save = '../trainedClassifier/'):
        """
        Description: Constructor for Label Classier 
        Input:  filename name of the file
                data to save
        Output: Return nothing
        """
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
    
    def trainClassifier(self, X_train, y_train,loadClassifier = False, saveToFile = False):
        """
        Description: Constructor for Label Classier 
        Input:  X_train training documents
                y_train labels for training documents
        Output: Nothing
        """
        logging.info("> training classifier")
        voting = None
        if loadClassifier == True:
            try:
                self.trainedEstimator = joblib.load(self.fileLocation)
                voting = loadClassifier.getVotingClassifier()
            except:
                raise("load voting classifier failed")
                
        else:
            self.trainedEstimator = VotingClassifier(self.estimators, voting='hard')
            voting = self.trainedEstimator.fit_transform(X_train, y_train) # test our model on the test data
            if saveToFile == True:
                joblib.dump(self.trainedEstimator , self.fileLocation, compress=9)
                joblib.dump(voting, '../trainedClassifier/VotingClassifier',compress=9)
                logging.info("> dumped Classifier: {}".format(self.fileLocation))
        self.trainKernelApproxSvgOnVoting(voting, y_train)

    def predict(self, X_test):
        """
        Description: Method labels data
        Input:  X_test data
        Output: Trained estimator
        """
        logging.info("> predicting")
        return self.trainedEstimator.predict(X_test)

    def generateFilename(self, folder = '../trainedClassifier/'):
        """
        Description: Method generates Filename for classifier
        Input:  Nothing
        Output: Filename as string
        """
        return "{}ensembleClassifier_{}-{}.joblib.pkl".format(folder, self.category[0], self.category[1])

    def accuracy(self, X_test, y_test, predicted):
        """
        Description: Methods plots the accuracy of the trained classifier
        Input:  X_test test documents
                y_test labels for the test documents
                predicted 
        Output: None
        """
        if self.trainedEstimator == None:
            raise AssertionError("Classifier has not been trained yet")
        logging.info("\n ->> ensemble-score:{}\n".format(np.mean(predicted == y_test)))
        plot_confusion_matrix(self.trainedEstimator, X_test, y_test, normalize="all",display_labels=[self.category[0],self.category[1]])
        plt.show()
    
    def trainKernelApproxSvgOnVoting(self, X_predicted, y):
        """
        Description: Train kernel for classifier
        Input:  X_predicted training data
                y_test labels 
        Output: Filename as string
        """
        logging.info("training stacking classifier")
        self.rbfKernel = RBFSampler(gamma=1, random_state=1)
        X_features = self.rbfKernel.fit_transform(X_predicted)
        self.stackingEstimator = SGDClassifier(max_iter=1000)
        self.stackingEstimator.fit(X_features, y)
        logging.info("stacking-classifier: " + str(self.stackingEstimator.score(X_features, y)))
    
    def stackingPrediction(self, X_test):
        """
        Description: Method predict stacking 
        Input:  X_test training documents
        Output: Return Prediction
        """
        voting = self.trainedEstimator.transform(X_test)
        influencedVoting = self.rbfKernel.transform(voting)
        return self.stackingEstimator.predict(influencedVoting)
