import matplotlib.pyplot as plt
import numpy
import nltk
import logging

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

import json
import joblib
import vectorizer
import file_manipulation
import configuration

config = configuration.Configuration()

class DataPreprocessor(vectorizer.Vectorizer):
    """This class is used to preprocess the data before training.

    Args:
        vectorizer (Vectorizer): The vectorizer is used for creating a feature vector.
    """
    def __init__(self):
        """This is the constructor to create a DataPreprocessor object.
        """
        super().__init__()
        self.reverseData:list = []

    def train_test_split(self, X:numpy.ndarray, y:numpy.ndarray) -> tuple:
        """This method is used to split the documents into a training and testing array.

        Args:
            X (numpy.ndarray): A list of documents (strings).
            y (numpy.ndarray): The description for the documents. Either a 1 or 0, shows whether the document uses label 1 or 0.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) The training data, testing data, solutions.
        """
        if not X.size:
            raise("Parameter 'X' of type numpy.ndarray is empty!")
        if not y.size:
            raise("Parameter 'y' of type numpy.ndarray is empty!")
        trainingPercentage:float = config.getValueFromConfig("trainingConstants trainingPercentage")
        numpy.random.seed(config.getValueFromConfig("trainingConstants randomSeed"))
        # 70% for training, 30% for testing - no cross validation yet
        threshold:int = int(trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx:numpy.ndarray = numpy.random.permutation(X.shape[0])
        # just normal array slices

        X_vectorrized:scipy.sparse.csr.csr_matrix = self.Vecotrizer.transform(X)
        X_train:numpy.ndarray = X_vectorrized[rnd_idx[:threshold]]
        X_test:numpy.ndarray = X_vectorrized[rnd_idx[threshold:]]
        logging.info("training on: {}% == {} documents\ntesting on: {} documents".format(
            trainingPercentage, threshold, X.shape[0]-threshold))
        #logging.info(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]])
        # logging.info(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]]
        # rnd_idx = reverseData[i][1]
        self.reverseData.append(rnd_idx)

        y_train:numpy.ndarray = y[rnd_idx[:threshold]]
        y_test:numpy.ndarray = y[rnd_idx[threshold:]]
        # create feature vectors TODO maby store the create vector func
        return X_train, X_test, y_train, y_test

    def getTrainingAndTestingData(self) -> tuple:
        """This method returns the training and testing data for multiple categories.
        It yields the specific data for each category tuple.

        Yields:
            Iterator[tuple]: (X_train, X_test, y_train, y_test) The training data, testing data, solutions for the specific categories.
        """
        for cat in config.getValueFromConfig("categories"):
            yield self.trainingAndTestingDataFromCategory(cat)
    
    def trainingAndTestingDataFromCategory(self, categorieArray:list) -> tuple:
        """This method is used for loading the training and testing data from a specific categorie as well as creating the training and testing data.

        Args:
            categorieArray (list): A array of categories i.e. [("bug","enhancement"), ("doku", "api", "bug")].

        Returns:
            tuple: [description] Returns (X_train, X_test, y_train, y_test) the training data, testing data, solutions for the specific categories.
        """
        if not categorieArray:
            raise("Parameter 'categorieArray' of type list is empty!")
        logging.info("train+testData")
        # input: [a,b,...,c] a wird gegen b,...,c getestet.
        path:str = "{}/{}.json".format(config.getValueFromConfig("issueFolder"), categorieArray[0])
        classAsize:int = self.openFile(path).shape[0]
        dataPerClassInB:int = (int)(classAsize/(len(categorieArray)-1))
        logging.info("dataPerClassInB: {}".format(dataPerClassInB))
        classB:numpy.ndarray = numpy.array([])
        for category in categorieArray[1:]:
            classB = numpy.append(classB, self.getRandomDocs(category, dataPerClassInB))
            logging.info("classB size = {} Byte".format(classB.itemsize))

        classBsize:int = classB.shape[0]
        y:numpy.ndarray = numpy.ones(classBsize)
        # Important, A is appended after B, means X = [(b,...,n), a]
        if (classAsize > classBsize):
            y:numpy.ndarray = numpy.append(y, numpy.zeros(classBsize))
            X:numpy.ndarray = numpy.append(self.getRandomDocs(categorieArray[0], classBsize), classB)
        else:
            y:numpy.ndarray = numpy.append(y, numpy.zeros(classAsize))  # A might be smaller
            X:numpy.ndarray = numpy.append(self.openFile(path), classB)
        return self.train_test_split(X, y)
