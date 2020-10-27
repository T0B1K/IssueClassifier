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

class AntMapPreprozessor(vectorizer.Vectorizer):
    """This class is used to get a different kind of Preprozessor, the so called AntMapPreprozessor.
    It is only used for sanity checking purposes, because it creates an so called "ant map" for trainings evaluation purposes 

    Args:
        vectorizer (Vectorizer): The Vectorizer is used for creating a feature vector
    """
    def __init__(self):
        """This is the constructor of the class and it is used to create an AntMapPreprozessor object.
        """
        super().__init__()
        self.reverseData:list = []
        self.labelClasses:list = file_manipulation.FileManipulation.values["labelClasses"]
        self.categories:list = file_manipulation.FileManipulation.values["categories"]

    def loadDataFromClasses(self) -> list:
        """This method is used to load the (data/documents) from the label classes

        Returns:
            list: loaded documents from corresponding labels provided in the config file
        """

        listOfDocuments:list = []
        for lblClass in self.labelClasses:
            path:str = "{}/{}.json".format(file_manipulation.FileManipulation.values["issueFolder"], lblClass)
            tmp:numpy.ndarray = self.openFile(path)
            listOfDocuments.append(tmp)
        return listOfDocuments

    def dataCategorie(self, documents: list) -> tuple:
        """This method is used to creating document pairs for label pairs i.e. "bug" vs "enhancement"

        Args:
            documents (list): list of documents for each label procided in the config file

        Yields:
            Iterator[tuple]: (catA, catB) the name of the categiries, (X, y) the list of documents as well as the solutions, which label each document has
        """
        
        for name1, name2 in self.categories:
            idx1:int = self.labelClasses.index(name1)
            idx2:int = self.labelClasses.index(name2)
            X1:numpy.ndarray = documents[idx1]
            X2:numpy.ndarray = documents[idx2]
            minLen:int = min(len(X1), len(X2))
            X:numpy.ndarray = numpy.append(X1[:minLen], X2[:minLen])
            y:numpy.ndarray = numpy.append(numpy.zeros(minLen), numpy.ones(minLen))
            yield (name1, name2), (X, y)

    def train_test_split(self, X:numpy.ndarray, y:numpy.ndarray) -> tuple:
        """This method shuffels the documents using a random permutatoin, as well as
        splitting the document array into an training and an testing part using the treshold provided in the config file

        Args:
            X (numpy.ndarray): The list of trainings documents
            y (numpy.ndarray): The corresponding solutions if the specific document belongs either to label 1 or label 0

        Returns:
            tuple:  X_train numpy.ndarray[String]    The training documents
                    X_test  numpy.ndarray[String]    The testing documents
                    y_train numpy.ndarray[String]    The train document solutions
                    y_test  numpy.ndarray[String]    the test document solutions
        """
                
        numpy.random.seed(file_manipulation.FileManipulation.values["trainingConstants"]["randomSeed"])
        trainingPercentage:float = file_manipulation.FileManipulation.values["trainingConstants"]["trainingPercentage"]
        threshold:int = int(trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx:numpy.ndarray = numpy.random.permutation(X.shape[0])
        # just normal array slices
        X_unvectorized_train:numpy.ndarray = X[rnd_idx[:threshold]]
        X_unvectorized_test:numpy.ndarray = X[rnd_idx[threshold:]]

        #logging.info(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]])
        # logging.info(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]]
        # rnd_idx = reverseData[i][1]
        self.reverseData.append(rnd_idx)

        y_train:numpy.ndarray = y[rnd_idx[:threshold]]
        y_test:numpy.ndarray = y[rnd_idx[threshold:]]
        X_train, X_test = self.createFeatureVectors(
            X_unvectorized_train, X_unvectorized_test)
        return X_train, X_test, y_train, y_test

    def findDocument(self, permutedIdx:list, category:list, justReturnIndex:bool=False) -> list:
        """This method is used for finding a specific document

        Args:
            permutedIdx (list): The permutation index from the specific document
            category (list):  The corresponding categories
            justReturnIndex (bool, optional): Whether just the index should be returned or also the text. Defaults to False.

        Returns:
            list: The specific returnvalues, where to find the documents
        """
        
        docs:list = self.loadDataFromClasses()
        X, y = next(self.dataCategorie(docs))[1]
        catIdx:int = self.categories.index(category[0])
        permutation:list = self.reverseData[catIdx]
        returnvalue:list = [permutation[permIdx] for permIdx in permutedIdx]
        if not justReturnIndex:
            returnvalue:list = [X[idx] for idx in returnvalue]
        return returnvalue

    def getTrainingAndTestingData(self) -> tuple:
        """This method returns the training and testing data for the specific categories

        Yields:
            Iterator[tuple]: The splitted training and testing data
                X_train numpy.ndarray[String]    The training documents
                X_test  numpy.ndarray[String]    The testing documents
                y_train numpy.ndarray[String]    The train document solutions
                y_test  numpy.ndarray[String]    the test document solutions
        """
        
        docs:list = self.loadDataFromClasses()
        for i, j in self.dataCategorie(docs):
            logging.debug(i)
            yield self.train_test_split(j[0], j[1])

    def createAntMap(self, tmpIncList:list, category:list, classificationMistakes:list, antmap:list):
        """This method is used for creating an antmap and saving it to a file

        Args:
            tmpIncList (list): lists all indices of wrong classified issues
            category (list): the corresbonding categories
            classificationMistakes (list): prepared list of wrong classified issues
            antmap (list): empty antmap
        """

        wrongClassifiedDocumentIdx:list = self.findDocument(
            tmpIncList, category, justReturnIndex=True)
        wrongClassifiedDocuments:list = self.findDocument(tmpIncList, category)
        for idx in wrongClassifiedDocumentIdx:
            antmap[idx]:str = "✖"
        antmap[(int)(len(antmap)/2)] += "\n\n----  ▲ {} --------- {} ▼  ----\n\n".format(
            category[0][0], category[0][1])
        nameAddon:str = "_{}-{}".format(category[0][0], category[0][1])

        self.saveWrongClassifiedToFile("wrong_classified{}.json".format(
            nameAddon), zip(classificationMistakes, wrongClassifiedDocuments))
        self.saveAntmapToFile("new_antmap{}.txt".format(
            nameAddon), " ".join(antmap))

    def prepareAntMap(self, Xpredicted:numpy.ndarray, yTest:numpy.ndarray, Xtrain:numpy.ndarray, category:list):
        """This method is used for preparing the antmap data.

        Args:
            Xpredicted (numpy.ndarray): The predicted label
            yTest (numpy.ndarray): the testlabels
            Xtrain (numpy.ndarray): the trainings data
            category (list): the corresbonding categories

        Raises:
            AttributeError: If the prediction array shape doesn't match the testing array shape
        """
        
        if not Xpredicted.shape == yTest.shape:
            raise AttributeError("prediction shape doesn't match test shape")
        lenTrain:int = Xtrain.shape[0]
        lenPred:int = Xpredicted.shape[0]
        antmap:list = self.antMapPreprocessing(lenTrain, lenPred, category)

        # (1,1) || (0,0) => 0; (1,0)=> 1 = predicted category[1] but was category[0]; (0,1)=>-1 = pred. cat.[0] but was cat.[1]
        classification:numpy.ndarray = Xpredicted-yTest
        tmpIncList:list = []
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

    def antMapPreprocessing(self, lenTrain:int, lenPred:int, category:tuple) -> list:
        """This method is used preporcessing the antmap array i.e. how it should be printed

        Args:
            lenTrain (int): length of the trainingsdata
            lenPred (int): length of the predicted data
            category (tuple): category labels

        Returns:
            list: The partly finished antmap array
        """ 
                
        antmap:list = ["_"]*(lenPred+lenTrain)
        # train = [0, treshold]; test = (treshold, inf] (so we have to add lenPred onto the idx to get the testIdx)
        testedPart:list = list(map(lambda x: x+lenPred, range(lenPred)))
        classified:list = self.findDocument(testedPart, category, justReturnIndex=True)
        for x in classified:
            antmap[x]:str = "✓"
        return antmap

    def getAllDocs(self) -> numpy.ndarray:
        """This method is used to get all the documents from all the labels, which are loaded

        Returns:
            numpy.ndarray: the documents to the corresponding labels
        """
        
        listOfDocuments:numpy.ndarray = numpy.empty()
        for lblClass in self.labelClasses:
            path:str = "{}/{}.json".format(file_manipulation.FileManipulation.values["issueFolder"], lblClass)
            tmp:numpy.ndarray = self.openFile(path)
            listOfDocuments:numpy.ndarray = numpy.append(listOfDocuments, tmp)
        return listOfDocuments
