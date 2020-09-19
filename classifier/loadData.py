import matplotlib.pyplot as plt
import numpy as np
import nltk

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix

from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')
import json
import joblib


class DataPreprocessor:

    def __init__(self, labelClasses, categories, loadVec=True, saveVec=False, trainingPercentage=0.7, ngram=(1, 2), stripAccents=None, stopWords=None,
                  outputFolder="../auswertungen"):
        self.trainingPercentage = trainingPercentage
        self.reverseData = []
        self.randPerm = []
        self.labelClasses = labelClasses
        self.categories = categories
        self.folderName = "../documents"
        self.outputFolder = outputFolder
        self.Vecotrizer = self.prepareVectorizer(loadVec, saveVec, stripAccents, ngram, stopWords)

    # This method opens a file and returns all the documents

    def openFile(self, filename, elementcount=5000):
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = np.array(list(map(lambda entry: entry["text"], json.loads(data))))
        return documents[:elementcount]

    # this is a stemmer
    def stemmer(self, text):
        return [PorterStemmer().stem(token) for token in text]

    # this is a lemmatizer
    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(token) for token in text]

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
    """
    # load data from label categories
    def loadDataFromClasses(self, consoleOutput=True):
        listOfDocuments = []
        for lblClass in self.labelClasses:
            if consoleOutput:
                print(lblClass)
            path = "{}/{}.json".format(self.folderName, lblClass)
            tmp = self.openFile(path)
            listOfDocuments.append(tmp)
            if consoleOutput:
                print("> {} issues in {}".format(len(tmp), lblClass))
        return listOfDocuments

    def dataCategorie(self, documents, output=True):
        for name1, name2 in self.categories:
            idx1 = self.labelClasses.index(name1)
            idx2 = self.labelClasses.index(name2)
            X1 = documents[idx1]
            X2 = documents[idx2]
            minLen = min(len(X1), len(X2))
            if output:
                print("minlen: {}".format(minLen))
            X = np.append(X1[:minLen], X2[:minLen])
            y = np.append(np.zeros(minLen), np.ones(minLen))
            yield (name1, name2), (X, y)
    
    #TODO adapt to new changes
    def findDocument(self, permutedIdx, category, justReturnIndex=False):
        docs = self.loadDataFromClasses(consoleOutput=False)
        X, y = next(self.dataCategorie(docs, output=False))[1]
        catIdx = self.categories.index(category[0])
        permutation = self.reverseData[catIdx]
        returnvalue = [permutation[permIdx] for permIdx in permutedIdx]
        if not justReturnIndex:
            returnvalue = [X[idx] for idx in returnvalue]
        return returnvalue

    def createAntMapAndDocumentView(self, Xpredicted, yTest, Xtrain, category):
        # idee: erst alles auf trainingsdata also "."; danach predicted len auf "-" und dann die fehler auf "X"
        if not Xpredicted.shape == yTest.shape:
            raise AttributeError("prediction shape doesn't match test shape")
        lenTrain = Xtrain.shape[0]
        lenPred = Xpredicted.shape[0]
        antmap = self.antmapPreprocessing(lenTrain, lenPred, category)

        # (1,1) || (0,0) => 0; (1,0)=> 1 = predicted category[1] but was category[0]; (0,1)=>-1 = pred. cat.[0] but was cat.[1]
        classification = Xpredicted-yTest
        tmpInxList = []
        classificationMistakes = []
        for i in range(lenPred):
            if classification[i] == 0:
                continue
            classAs = category[0][0]
            if classification[i] == 1:
                classAs = category[0][1]
            classificationMistakes.append(classAs)
            tmpInxList.append(i+lenTrain)

        wrongClassifiedDocumentIdx = self.findDocument(
            tmpInxList, category, justReturnIndex=True)
        wrongClassifiedDocuments = self.findDocument(tmpInxList, category)
        for idx in wrongClassifiedDocumentIdx:
            antmap[idx] = "✖"
        antmap[(int)(len(antmap)/2)] += "\n\n----  ▲ {} --------- {} ▼  ----\n\n".format(
            category[0][0], category[0][1])
        nameAddon = "_{}-{}".format(category[0][0], category[0][1])

        self.saveWrongClassifiedToFile("newWrongClassifiedDocuments{}.json".format(
            nameAddon), zip(classificationMistakes, wrongClassifiedDocuments))
        self.saveAntmapToFile("newAntmap{}.txt".format(
            nameAddon), " ".join(antmap))

    def antmapPreprocessing(self, lenTrain, lenPred, category):
        antmap = ["_"]*(lenPred+lenTrain)
        # train = [0, treshold]; test = (treshold, inf] (so we have to add lenPred onto the idx to get the testIdx)
        testedPart = list(map(lambda x: x+lenPred, range(lenPred)))
        classified = self.findDocument(
            testedPart, category, justReturnIndex=True)
        for x in classified:
            antmap[x] = "✓"
        return antmap

    def saveWrongClassifiedToFile(self, filename, data):
        path = self.outputFolder+"/"+filename
        print(">\tsaving Wrong Classified Texts in {}".format(path))
        f = open(path, "w", encoding='utf-8', errors='ignore')
        jsonData = []
        for classified, document in data:
            jsonData.append({
                "classified_as": classified,
                "text": document
            })
            # convert into JSON:
        f.write(json.dumps(jsonData))
        f.close()

    def saveAntmapToFile(self, filename, data):
        path = self.outputFolder+"/"+filename
        print(">\tsaving antmap in {}".format(path))
        f = open(path, "w", encoding='utf-8')
        f.write(data)
        f.close()
    """
    def prepareVectorizer(self, loadVec, saveVec, stripAccents, ngram, stopWords):
        Vecotrizer = None
        if loadVec == True:
            try:
                Vecotrizer = joblib.load('../vectorizer.vz', )
                return Vecotrizer
            except:
                print("Vec could not be loaded")
                raise
                # prepareVectorizer(False,False)
        else:
            train_Data = self.getSplitedDocs(4000)
            Vecotrizer = TfidfVectorizer(tokenizer=None,
                                         strip_accents=stripAccents, lowercase=None, ngram_range=ngram,
                                         stop_words=stopWords,
                                         min_df=2)
            Vecotrizer.fit_transform(train_Data)
            if saveVec == True:
                joblib.dump(Vecotrizer, '../vectorizer.vz', compress=9)
            return Vecotrizer

    def getRandomDocs(self, label, elementcount):
        path = "{}/{}.json".format(self.folderName, label)
        data = self.openFile(path, elementcount)
        self.randPerm.append(np.random.permutation(data.shape[0]))
        print(self.randPerm[-1])
        return data[self.randPerm[-1]]

    def getSplitedDocs(self,sampleSize):
        length = len(self.labelClasses)
        docCount = round (sampleSize / length)
        docs = np.empty(0)
        for label in self.labelClasses:
            print("docs size: {} Byte".format(docs.itemsize))
            docs = np.append(self.getRandomDocs(label, docCount),docs)
        return docs
        
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
