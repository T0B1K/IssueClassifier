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

    def __init__(self,labelClasses, categories, loadVec = True ,saveVec = False , trainingPercentage=0.7, ngram = (1,2), stripAccents=None,stopWords=None, 
        numberToWordMapping = None, outputFolder="../auswertungen"):
        self.trainingPercentage = trainingPercentage
        self.ngram = ngram
        self.stripAccents = stripAccents
        self.stopWords = stopWords
        self.numberToWordMapping = numberToWordMapping
        self.reverseData = []
        self.labelClasses = labelClasses
        self.categories = categories
        self.folderName = "../documents"
        self.outputFolder = outputFolder
        self.Vecotrizer = self.prepareVectorizer(loadVec, saveVec)
        

    # This method opens a file and returns all the documents
    def openFile(self, filename):
        with open(filename, "r") as file:
            data = file.read()
        # we just take all the "text" from the JSON
        documents = list(map(lambda entry: entry["text"], json.loads(data)))
        return np.array(documents[:7000])

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

    # this is a stemmer
    def stemmer(self, text):
        return [PorterStemmer().stem(token) for token in text]

    # this is a lemmatizer
    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(token) for token in text]

    # This method is used to convert the documents to actual numbers
    # TODO maybe store
    # It returns the training data normalized to tfidf and the vectorized test data
    def createFeatureVectors(self, X_train_documents, X_test_documents):
        #the vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
        X_train_vectorized = self.Vecotrizer.transform(X_train_documents)    
        X_test_vectorized = self.Vecotrizer.transform(X_test_documents) #vectorisation
        return X_train_vectorized, X_test_vectorized

    def train_test_split(self, X, y):
        np.random.seed(2020)
        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.trainingPercentage*X.shape[0])
        # this is a random permutation
        rnd_idx = np.random.permutation(X.shape[0])
        # just normal array slices
        X_unvectorized_train = X[rnd_idx[:threshold]]
        X_unvectorized_test = X[rnd_idx[threshold:]]

        print("training on: {}% == {} documents\ntesting on: {} documents".format(self.trainingPercentage, threshold, X.shape[0]-threshold))
        #print(X_unvectorized_test[3] == X[rnd_idx[3+X_unvectorized_train.shape[0]]]) 
        #print(rnd_idx)                #mapping X_train[idx] = X[ rnd_idx[idx]] 
        self.reverseData.append(rnd_idx)                        # rnd_idx = reverseData[i][1]

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]
        # create feature vectors TODO maby store the create vector func
        X_train, X_test = self.createFeatureVectors(
            X_unvectorized_train, X_unvectorized_test)
        return X_train, X_test, y_train, y_test

    def findDocument(self, permutedIdx, category, justReturnIndex=False):
        docs = self.loadDataFromClasses(consoleOutput=False)
        X, y = next(self.dataCategorie(docs, output=False))[1]
        catIdx = self.categories.index(category[0])
        permutation = self.reverseData[catIdx]
        returnvalue = [permutation[permIdx] for permIdx in permutedIdx]
        if not justReturnIndex:
            returnvalue = [X[idx] for idx in returnvalue]
        return returnvalue

    def getTrainingAndTestingData(self, labelClasses, categories):
        self.labelClasses = labelClasses
        self.categories = categories
        docs = self.loadDataFromClasses()
        for i, j in self.dataCategorie(docs):
            print(i)
            yield self.train_test_split(j[0], j[1])

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
    
    def prepareVectorizer(self, loadVec,saveVec):
        Vecotrizer = None
        if loadVec == True:
            try:
                Vecotrizer = joblib.load('../vectorizer.vz', )
                return Vecotrizer
            except:
                print("Vec could not be loaded")
                raise
                #prepareVectorizer(False,False)
        else:
            train_Data = self.getAllDocs()
            Vecotrizer = TfidfVectorizer(tokenizer=None,\
                strip_accents=self.stripAccents,lowercase = None ,ngram_range=self.ngram,
                stop_words=self.stopWords,
                min_df=2)
            Vecotrizer.fit_transform(train_Data)
            if saveVec == True:
                joblib.dump(Vecotrizer, '../vectorizer.vz' ,compress = 9)
            return Vecotrizer
    
    def getAllDocs(self):
        listOfDocuments = np.empty(0)
        for lblClass in self.labelClasses:
            path = "{}/{}.json".format(self.folderName, lblClass)
            tmp = self.openFile(path)
            listOfDocuments = np.append(listOfDocuments,tmp)
        print(listOfDocuments.shape)
        return listOfDocuments

    def pascalFunc(self,lable, elementcount):
        path = "{}/{}.json".format(self.folderName, label)
        tmp = self.openFile(path)
        rnd = np.random.permutation(tmp)
        min = min(len(rnd),elementcount)
        rnd = rnd[:min]
        return rnd

    def onORother(self, labelClass,sampleSize):
        lenghts = np.empty()
        for element in filteredLabelClasse:
            path = "{}/{}.json".format(self.folderName, element)
            tmp = self.openFile(path)
            lenghts = lenghts.append((len(temp),element))
        minVal = min(for tpl[0] in lenghts)
        #UNFINISHED DO NOT USE 












    
    

    

