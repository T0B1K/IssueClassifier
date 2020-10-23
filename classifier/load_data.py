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

"""
Description: This class is used to preprocess the data 
"""
class DataPreprocessor(vectorizer.Vectorizer):
    def __init__(self, labelClasses, categories, loadVec=True, saveVec=False):
        super().__init__(labelClasses)
        self.trainingPercentage = file_manipulation.FileManipulation.values["trainingPercentage"]
        self.labelClasses = labelClasses
        self.categories = categories
        self.reverseData = []
        self.randPerm = []

    def train_test_split(self, X, y):
        """
        Description: This method is used to split the documents into a training and testing array
        Input X :List[String]       The documents
            y :List[String]       The corresponding label { 0, 1 }
        """
        numpy.random.seed(file_manipulation.FileManipulation.values["randomSeed"])
        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx = numpy.random.permutation(X.shape[0])
        # just normal array slices

        X_vectorrized = self.Vecotrizer.transform(X)
        X_train = X_vectorrized[rnd_idx[:threshold]]
        X_test = X_vectorrized[rnd_idx[threshold:]]
        logging.info("training on: {}% == {} documents\ntesting on: {} documents".format(
            self.trainingPercentage, threshold, X.shape[0]-threshold))
        #logging.info(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]])
        # logging.info(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]]
        # rnd_idx = reverseData[i][1]
        self.reverseData.append(rnd_idx)

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]
        # create feature vectors TODO maby store the create vector func
        return X_train, X_test, y_train, y_test

    def getTrainingAndTestingData2(self):
        """
        Description: This method returns the training and testing data for specified categories
        """
        for cat in self.categories:
            yield self.trainingAndTestingDataFromCategory(cat)
    
    def trainingAndTestingDataFromCategory(self, categorieArray):
        """
        Description: This method loads the training and testing data from specific categories
        Input:  categorieArray :List[String] i.e. [("bug","enhancement"), ("doku", "api", "bug")]
        Output: List[String], List[String]      returns the trainig and testing data
        """
        logging.info("train+testData")
        # input: [a,b,...,c] a wird gegen b,...,c getestet.
        path = "{}/{}.json".format(self.folderName, categorieArray[0])
        classAsize = self.openFile(path).shape[0]
        # TODO free memory
        dataPerClassInB = (int)(classAsize/(len(categorieArray)-1))
        logging.info("dataPerClassInB: {}".format(dataPerClassInB))
        classB = numpy.array([])
        for category in categorieArray[1:]:
            classB = numpy.append(classB, self.getRandomDocs(category, dataPerClassInB))
            logging.info("classB size = {} Byte".format(classB.itemsize))

        classBsize = classB.shape[0]
        y = numpy.ones(classBsize)
        # Important, A is appended after B, means X = [(b,...,n), a]
        if (classAsize > classBsize):
            y = numpy.append(y, numpy.zeros(classBsize))
            X = numpy.append(self.getRandomDocs(categorieArray[0], classBsize), classB)
        else:
            y = numpy.append(y, numpy.zeros(classAsize))  # A might be smaller
            X = numpy.append(self.openFile(path), classB)
        return self.train_test_split(X, y)
