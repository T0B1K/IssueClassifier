import matplotlib.pyplot as plt
import numpy
import nltk
import logging

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix

from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')  #if you are running the lemmatizer for the first time, you need this
import json
import joblib

import vectorizer
import file_manipulation

"""
This class is used to get a different AntMapPreprozessor - to generate an antmap
It is only used for sanity checking purposes
"""
class AntMapPreprozessor(vectorizer.Vectorizer):
    def __init__(self, labelClasses, categories):
        """
        Description: This is the constructor of the class AntMapPreprozessor
        Input:  labelClasses List[String]                   the classes for the labels ["bug", "doku", "api", "enhancement"]
                categories   List[Tuple(String, String)]    the categories i.e. [("bug","enhancement"), ("doku", "api")]
        """
        super().__init__(labelClasses)
        self.reverseData = []
        self.labelClasses = labelClasses
        self.categories = categories
        self.trainingPercentage = file_manipulation.FileManipulation.values["trainingPercentage"]

    def loadDataFromClasses(self, consoleOutput=True):
        """
        Description: This method is used to load the (data/documents) from the label classes
        Input: consoleOutput : Boolean (optional) (true default) determines, wether an informative console output should be logging.infoed or not
        Output: List[String] loaded documents from corresponding files
        """
        listOfDocuments = []
        for lblClass in self.labelClasses:
            if consoleOutput:
                logging.debug(lblClass)
            path = "{}/{}.json".format(self.folderName, lblClass)
            tmp = self.openFile(path)
            listOfDocuments.append(tmp)
            if consoleOutput:
                logging.debug("> {} issues in {}".format(len(tmp), lblClass))
        return listOfDocuments

    def dataCategorie(self, documents, output=True):
        """
        Description: This method is used to creating document pairs for label pairs i.e. "bug" vs "enhancement"
        Input:  List[List[String]] - list of documents for each label
        Output: returns the document lists to the categories
        """
        for name1, name2 in self.categories:
            idx1 = self.labelClasses.index(name1)
            idx2 = self.labelClasses.index(name2)
            X1 = documents[idx1]
            X2 = documents[idx2]
            minLen = min(len(X1), len(X2))
            if output:
                logging.debug("minlen: {}".format(minLen))
            X = numpy.append(X1[:minLen], X2[:minLen])
            y = numpy.append(numpy.zeros(minLen), numpy.ones(minLen))
            yield (name1, name2), (X, y)

    def train_test_split(self, X, y):
        """
        Description: This method is used for creating a random permutation, as well as splitting the document array into an training and an testing part using the treshold
        Input:  X List[String]    The list of documents
                y List[String]    The list of labels for the specifc documents { 0, 1 }
        Output: X_train List[String]    The training documents
                X_test  List[String]    The testing documents
                y_train List[String]    The train document solutions
                y_test  List[String]    the test document solutions
        """
        numpy.random.seed(file_manipulation.FileManipulation.values["randomSeed"])
        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx = numpy.random.permutation(X.shape[0])
        # just normal array slices
        X_unvectorized_train = X[rnd_idx[:threshold]]
        X_unvectorized_test = X[rnd_idx[threshold:]]

        #logging.info(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]])
        # logging.info(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]]
        # rnd_idx = reverseData[i][1]
        self.reverseData.append(rnd_idx)

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]
        # create feature vectors TODO maby store the create vector func
        X_train, X_test = self.createFeatureVectors(
            X_unvectorized_train, X_unvectorized_test)
        return X_train, X_test, y_train, y_test

    def findDocument(self, permutedIdx, category, justReturnIndex=False):
        """
        Description: This method is used for finding a specific document
        Input:  permutedIdx     The permutation index from the specific document
                The category    The corresponding categories
                justReturnIndex :Boolean (default false) just return the file
        Output: The specific returnvalues, where to find the documents
        """
        docs = self.loadDataFromClasses(consoleOutput=False)
        X, y = next(self.dataCategorie(docs, output=False))[1]
        catIdx = self.categories.index(category[0])
        permutation = self.reverseData[catIdx]
        returnvalue = [permutation[permIdx] for permIdx in permutedIdx]
        if not justReturnIndex:
            returnvalue = [X[idx] for idx in returnvalue]
        return returnvalue

    def getTrainingAndTestingData(self, labelClasses, categories):
        """
        Description: This method returns the training and testing data to the specific categories
        Input:  labelClasses List[String], categories
        Output: returns the splitted training and testing data
        """
        self.labelClasses = labelClasses
        self.categories = categories
        docs = self.loadDataFromClasses()
        for i, j in self.dataCategorie(docs):
            logging.debug(i)
            yield self.train_test_split(j[0], j[1])

    def createAntMap(self, tmpIncList, category, classificationMistakes, antmap):
        """
        Description: This method is used for creating an antmap and saving it to a file
        Input: tmpIncList             - lists all indices of wrong classified issues
               category               - the corresbonding categories
               classificationMistakes - prepared list of wrong classified issues
               antmap                 - empty antmap
        """
        wrongClassifiedDocumentIdx = self.findDocument(
            tmpIncList, category, justReturnIndex=True)
        wrongClassifiedDocuments = self.findDocument(tmpIncList, category)
        for idx in wrongClassifiedDocumentIdx:
            antmap[idx] = "✖"
        antmap[(int)(len(antmap)/2)] += "\n\n----  ▲ {} --------- {} ▼  ----\n\n".format(
            category[0][0], category[0][1])
        nameAddon = "_{}-{}".format(category[0][0], category[0][1])

        self.saveWrongClassifiedToFile("wrong_classified{}.json".format(
            nameAddon), zip(classificationMistakes, wrongClassifiedDocuments))
        self.saveAntmapToFile("new_antmap{}.txt".format(
            nameAddon), " ".join(antmap))

    def prepareAntMap(self, Xpredicted, yTest, Xtrain, category):
        """
        Description: This method is used for preparing the antmap data.
        Input:  Xpredicted - The predicted label
                yTest      - the testlabels
                Xtrain     - the trainings data
                category   - the corresbonding categories
        """
        # idee: erst alles auf trainingsdata also "."; danach predicted len auf "-" und dann die fehler auf "X"
        if not Xpredicted.shape == yTest.shape:
            raise AttributeError("prediction shape doesn't match test shape")
        lenTrain = Xtrain.shape[0]
        lenPred = Xpredicted.shape[0]
        antmap = self.antMapPreprocessing(lenTrain, lenPred, category)

        # (1,1) || (0,0) => 0; (1,0)=> 1 = predicted category[1] but was category[0]; (0,1)=>-1 = pred. cat.[0] but was cat.[1]
        classification = Xpredicted-yTest
        tmpIncList = []
        classificationMistakes = []
        for i in range(lenPred):
            if classification[i] == 0:
                continue
            classAs = category[0][0]
            if classification[i] == 1:
                classAs = category[0][1]
            classificationMistakes.append(classAs)
            tmpIncList.append(i+lenTrain)
        self.createAntMap(tmpIncList, category, classificationMistakes, antmap)

    def antMapPreprocessing(self, lenTrain, lenPred, category):
        """
        Description: This method is used preporcessing the antmap array i.e. how it should be logging.infoed
        Input:  lenTrain: int       length of the trainingsdata
                lenPred: int        length of the predicted data
                category: (string, string)  categorie labels
        """
        antmap = ["_"]*(lenPred+lenTrain)
        # train = [0, treshold]; test = (treshold, inf] (so we have to add lenPred onto the idx to get the testIdx)
        testedPart = list(map(lambda x: x+lenPred, range(lenPred)))
        classified = self.findDocument(
            testedPart, category, justReturnIndex=True)
        for x in classified:
            antmap[x] = "✓"
        return antmap

    def getAllDocs(self):
        """
        Description: This method is used to get all the documents from all the labels, which are loaded
        Output: List[ List[String] ] the documents to the corresponding labels
        """
        listOfDocuments = numpy.empty()
        for lblClass in self.labelClasses:
            path = "{}/{}.json".format(self.folderName, lblClass)
            tmp = self.openFile(path)
            listOfDocuments = numpy.append(listOfDocuments, tmp)
        return listOfDocuments
