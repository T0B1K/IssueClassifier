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


class LabelClassifier:
    """Class implemens various label Classifiers """

    def __init__(self, categoryToClassify: list, pretrained=None):
        """Constructor for Label Classier

        Args:
            categoryToClassify (list): The categories which should be classified
            pretrained ([type], optional): The pretrained classifier, if it exists. Defaults to None.
        """

        self.category: list = categoryToClassify
        self.estimators = estimators = [('MultinomialNB', MultinomialNB()),
                                        ('SGDClassifier', SGDClassifier(
                                            loss='modified_huber', penalty='l2', alpha=1e-3, random_state=100, max_iter=200)),
                                        ('sigmoidSVM', SVC(
                                            kernel='sigmoid', gamma=1.0)),
                                        ('RandomForest', RandomForestClassifier(
                                            200, bootstrap=False)),
                                        ('LogisticRegression', LogisticRegression(solver='sag', random_state=100))]
        self.trainedEstimator = pretrained
        self.fileLocation: str = self.generateFilename(
            file_manipulation.FileManipulation.values["classifier"]["path"]["saveFolder"])
        self.stackingEstimator = None
        self.rbfKernel = None

    def trainingClassifier(self, X_train: numpy.ndarray, y_train: numpy.ndarray):
        """This method trains the classifier

        Args:
            X_train (numpy.ndarray): training documents
            y_train (numpy.ndarray): labels for training documents
        """

        logging.info("> training classifier")
        voting = None
        if file_manipulation.FileManipulation.values["classifier"]["loadClassifier"] == True:
            try:
                self.trainedEstimator = joblib.load(self.fileLocation)
                voting = load_classifier.getVotingClassifier()
            except:
                raise("load voting classifier failed")

        else:
            self.trainedEstimator = VotingClassifier(
                self.estimators, voting='hard')
            voting = self.trainedEstimator.fit_transform(
                X_train, y_train)  # test our model on the test data
            if file_manipulation.FileManipulation.values["classifier"]["saveClassifier"] == True:
                joblib.dump(self.trainedEstimator,
                            self.fileLocation, compress=9)
                joblib.dump(
                    voting, '../classifier/trained_classifiers/voting_classifier', compress=9)
                logging.info(
                    "> dumped Classifier: {}".format(self.fileLocation))
        self.trainKernelApproxSvgOnVoting(voting, y_train)

    def predict(self, X_test: numpy.ndarray) -> numpy.ndarray:
        """This Method labels data by using the classifier to predict it

        Args:
            X_test (numpy.ndarray): data to be labeled

        Returns:
            numpy.ndarray: Trained estimator
        """

        logging.info("> predicting")
        return self.trainedEstimator.predict(X_test)

    def generateFilename(self, folder: str = '../trained_classifiers/') -> str:
        """This method generates the filename/path for the classifiers

        Args:
            folder (str, optional): The folder, where the filepath should lead to. Defaults to '../trained_classifiers/'.

        Returns:
            str: The filepath as a string
        """
        if len(self.category) == 3:
            return "{}ensembleClassifier_{}-{}-{}.joblib.pkl".format(folder, self.category[0], self.category[1], self.category[2])
        else:
            return "{}ensembleClassifier_{}-{}.joblib.pkl".format(folder, self.category[0], self.category[1])

    def accuracy(self, X_test: numpy.ndarray, y_test: numpy.ndarray, predicted: numpy.ndarray):
        """Methods plots the accuracy of the trained classifier

        Args:
            X_test (numpy.ndarray): The test documents
            y_test (numpy.ndarray): The corresbonding label for the dimmension
            predicted (numpy.ndarray): The predicted labels

        Raises:
            AssertionError: Raises this error if the estimator hasn't be trained first
        """
        if self.trainedEstimator == None:
            raise AssertionError("Classifier has not been trained yet")
        logging.info(
            "\n ->> ensemble-score:{}\n".format(numpy.mean(predicted == y_test)))
        plot_confusion_matrix(self.trainedEstimator, X_test, y_test, normalize="all", display_labels=[
                              self.category[0], self.category[1]])
        plt.show()

    def trainKernelApproxSvgOnVoting(self, X_predicted: numpy.ndarray, y: numpy.ndarray):
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
        logging.info("stacking-classifier: " +
                     str(self.stackingEstimator.score(X_features, y)))

    def stackingPrediction(self, X_test: numpy.ndarray) -> numpy.ndarray:
        """This method predicts using stacking

        Args:
            X_test (numpy.ndarray): The testing documents

        Returns:
            numpy.ndarray: stacking prediction
        """
        
        voting = self.trainedEstimator.transform(X_test)
        influencedVoting = self.rbfKernel.transform(voting)
        return self.stackingEstimator.predict(influencedVoting)
