import matplotlib.pyplot as plt
import numpy as np
import nltk

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

import json
import joblib
import vectorizer

class DataPreprocessor(vectorizer.Vectorrizer):
    def __init__(self, labelClasses, categories, trainingPercentage=0.7, loadVec=True, saveVec=False):
        super().__init__()
        self.trainingPercentage = trainingPercentage
        self.labelClasses = labelClasses
        self.categories = categories
        self.reverseData = []
        self.randPerm = []

    def train_test_split(self, X, y):
        np.random.seed(2020)
        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx = np.random.permutation(X.shape[0])
        # just normal array slices

        X_vectorrized = self.Vecotrizer.transform(X)
        X_train = X_vectorrized[rnd_idx[:threshold]]
        X_test = X_vectorrized[rnd_idx[threshold:]]
        print("training on: {}% == {} documents\ntesting on: {} documents".format(
            self.trainingPercentage, threshold, X.shape[0]-threshold))
        #print(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]])
        # print(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]]
        # rnd_idx = reverseData[i][1]
        self.reverseData.append(rnd_idx)

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]
        # create feature vectors TODO maby store the create vector func
        return X_train, X_test, y_train, y_test
        
    def getTrainingAndTestingData2(self):
        for cat in self.categories:
            yield self.trainingAndTestingDataFromCategory(cat)

    def trainingAndTestingDataFromCategory(self, categorieArray):
        print("train+testData")
        # input: [a,b,...,c] a wird gegen b,...,c getestet.
        path = "{}/{}.json".format(self.folderName, categorieArray[0])
        classAsize = self.openFile(path).shape[0]
        # TODO free memory
        dataPerClassInB = (int)(classAsize/(len(categorieArray)-1))
        print("dataPerClassInB: {}".format(dataPerClassInB))
        classB = np.array([])
        for category in categorieArray[1:]:
            classB = np.append(classB, self.getRandomDocs(category, dataPerClassInB))
            print("classB size = {} Byte".format(classB.itemsize))

        classBsize = classB.shape[0]
        y = np.ones(classBsize)
        # Important, A is appended after B, means X = [(b,...,n), a]
        if (classAsize > classBsize):
            y = np.append(y, np.zeros(classBsize))
            X = np.append(self.getRandomDocs(categorieArray[0], classBsize), classB)
        else:
            y = np.append(y, np.zeros(classAsize))  # A might be smaller
            X = np.append(self.openFile(path), classB)
        return self.train_test_split(X, y)
