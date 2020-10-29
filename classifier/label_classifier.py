import joblib
import matplotlib.pyplot as plt
import numpy

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import load_classifier

import logging
import file_manipulation
import configuration

config = configuration.Configuration()

class LabelClassifier:
    """Class implemens various label Classifiers """

    def __init__(self, categoryToClassify:list, pretrained = None):
        """Constructor for Label Classier

        Args:
            categoryToClassify (list): data to save
            pretrained ([type], optional): Pretrained classifier. Defaults to None.
        """
        if not categoryToClassify:
            raise("no categories to classify have been provided")
        self.category:list = categoryToClassify
        self.estimators = estimators=[('MultinomialNB', MultinomialNB()), \
        ('SGDClassifier', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)),
        ('sigmoidSVM', SVC(kernel='sigmoid', gamma=1.0)),
        ('RandomForest', RandomForestClassifier(200, bootstrap=False)),
        ('LogisticRegression',LogisticRegression(solver='sag',random_state=100))]
        self.trainedEstimator = pretrained
        self.fileLocation:str = self.generateFilename(config.getValueFromConfig("classifier path saveFolder"))
        self.stackingEstimator = None
        self.rbfKernel = None
    
    def trainingClassifier(self, X_train:numpy.ndarray, y_train:numpy.ndarray):
        """Constructor for Label Classier

        Args:
            X_train (numpy.ndarray): X_train training documents
            y_train (numpy.ndarray): y_train labels for training documents
        """
        if not X_train.size:
            raise("No X_train data was provided")
        if not y_train.size:
            raise("No y_train data was provided")
        logging.info("> training classifier")
        voting = None
        if config.getValueFromConfig("classifier loadClassifier") == True:
            try:
                self.trainedEstimator = joblib.load(self.fileLocation)
                voting = load_classifier.getVotingClassifier()
            except:
                raise("load voting classifier failed")
                
        else:
            self.trainedEstimator = VotingClassifier(self.estimators, voting='hard')
            voting = self.trainedEstimator.fit_transform(X_train, y_train) # test our model on the test data
            if config.getValueFromConfig("classifier saveClassifier") == True:
                joblib.dump(self.trainedEstimator , self.fileLocation, compress=9)
                joblib.dump(voting, '../classifier/trained_classifiers/voting_classifier',compress=9)
                logging.info("> dumped Classifier: {}".format(self.fileLocation))
        self.trainKernelApproxSvgOnVoting(voting, y_train)

    def predict(self, X_test:numpy.ndarray) -> numpy.ndarray:
        """Method labels data

        Args:
            X_test (numpy.ndarray): X_test data

        Returns:
            numpy.ndarray: Trained estimator prediction
        """
        if not X_test.size:
            raise("No test documents were provided")
        logging.info("> predicting")
        prediction = self.trainedEstimator.predict(X_test)
        assert prediction.size, "No documents were predicted"
        return prediction

    def generateFilename(self, folder = '../trained_classifiers/') -> str:
        """Method generates Filename for classifier

        Args:
            folder (str, optional): The folder path. Defaults to '../trained_classifiers/'.

        Returns:
            str: Filename as string
        """
        if folder == None:
            raise("No folder name was provided")
        if len(self.category) <2 or len(self.category) > 3:
            raise("To few or many categories")
        if len(self.category) == 3:
            return "{}ensembleClassifier_{}-{}-{}.joblib.pkl".format(folder, self.category[0],self.category[1],self.category[2])
        else:
            return "{}ensembleClassifier_{}-{}.joblib.pkl".format(folder, self.category[0],self.category[1])

    def accuracy(self, X_test:numpy.ndarray, y_test:numpy.ndarray, predicted:numpy.ndarray):
        """Methods plots the accuracy of the trained classifier

        Args:
            X_test (numpy.ndarray): The test documents
            y_test (numpy.ndarray): The results for the test documents
            predicted (numpy.ndarray): The predicted test values 

        Raises:
            AssertionError: This error is being thrown, if the classifier wasn't trained previousely
        """
        if not X_test.size:
            raise("X_test was empty")
        if not y_test.size:
            raise("y_test was empty")
        if not predicted.size:
            raise("predicted was empty")
        if self.trainedEstimator == None:
            raise AssertionError("Classifier has not been trained yet")
        logging.info("\n ->> ensemble-score:{}\n".format(numpy.mean(predicted == y_test)))
        plot_confusion_matrix(self.trainedEstimator, X_test, y_test, normalize="all",display_labels=[self.category[0],self.category[1]])
        plt.show()
    
    def trainKernelApproxSvgOnVoting(self, X_predicted:numpy.ndarray, y:numpy.ndarray):
        """Train kernel for classifier

        Args:
            X_predicted (numpy.ndarray): The prediction of the other classifiers.
            y (numpy.ndarray): The real labels.
        """
        if not X_predicted.size:
            raise("No X_predicted data was orovided")
        if not y.size:
            raise("No y data was provided")
        logging.info("training stacking classifier")
        self.rbfKernel = RBFSampler(gamma=1, random_state=1)
        X_features = self.rbfKernel.fit_transform(X_predicted)
        self.stackingEstimator = SGDClassifier(max_iter=config.getValueFromConfig("SGDClassifierIterations"))
        self.stackingEstimator.fit(X_features, y)
        logging.info("stacking-classifier: " + str(self.stackingEstimator.score(X_features, y)))
    
    def stackingPrediction(self, X_test: numpy.ndarray) -> numpy.ndarray:
        """This method predicts the result using another classifier - so called "stacking"

        Args:
            X_test (numpy.ndarray): The vectorized documents to test on. 

        Returns:
            numpy.ndarray: The prediction for the labels using stacking.
        """
        if not X_test.size:
            raise("No X_test data was provided")
        voting = self.trainedEstimator.transform(X_test)
        influencedVoting = self.rbfKernel.transform(voting)
        prediction = self.stackingEstimator.predict(influencedVoting)
        assert prediction.size
        return prediction
