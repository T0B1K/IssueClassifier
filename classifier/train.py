import matplotlib.pyplot as plt
import numpy as np
import nltk

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

from nltk.stem import WordNetLemmatizer, PorterStemmer
#nltk.download('wordnet')

import json 
import seaborn as sn
import pandas as pd

scores = []             #scores of the classifiers
X_test_documents = None #the texts of the test documents
documentTexts = None    #all the documents
trainingPercentage=0.7  #This method returns X_train, X_test, y_train, y_test, of which 70% are trainingdata and 30% for testing
permutation = None      #how the data was permuted

#data preprocessing parameters
ngram = (1,2)
lowerCase = True
stripAccents=None  #'ascii'
stopWords=None      #'english' | None

numberToWordMapping = None
tidvectorizer = None
listOfPropab = []

# This method opens a file and returns all the documents
def openFile(filename):
    with open(filename, "r") as file:
        data=file.read()
    jsonList = json.loads(data)
    documents = list(map(lambda entry: entry["text"], jsonList))
    print("data-count:{}".format(len(documents)))
    return np.array(documents[:1788])   #the number was chosen, so that the data sets are equal likely

def writeToFile(filename, data):
    f = open(filename, "w")
    jsonData = []
    for x in data:
        jsonData.append({
            "classified_as": x[0],
            "text": x[1].lower()
        })
        # convert into JSON:
    f.write(json.dumps(jsonData))
    f.close()

def train_test_split(X, y):
    global permutation, X_test_documents
    np.random.seed(2020)  # same seed for reproduceability
    threshold = int(trainingPercentage*X.shape[0])  #70% for training, 30% for testing - no cross validation yet
    rnd_idx = np.random.permutation(X.shape[0])     #this is a random permutation
    
    X_unvectorized_train = X[rnd_idx[:threshold]]                #just normal array slices
    X_unvectorized_test = X[rnd_idx[threshold:]]
    X_test_documents = X_unvectorized_test
    y_train = y[rnd_idx[:threshold]]
    y_test = y[rnd_idx[threshold:]]

    permutation = rnd_idx[threshold:]

    #vectorize the document arrays
    X_train, X_test = createFeatureVectors(X_unvectorized_train, X_unvectorized_test)
    return X_train, X_test, y_train, y_test

#This method loads the bugs and enhancements and returns the splitted data
def loadBugAndEnhancement():
    global documentTexts
    print("loading documents")
    X1 = openFile("doku.json")
    X2 = openFile("enhancement.json")
    y1 = [0]*len(X1)
    y2 = [1]*len(X2)
    documentTexts = np.append(X1,X2)
    y = np.append(y1,y2)
    print(len(X1), len(X2), len(documentTexts))

    return train_test_split(documentTexts, y)

#this method creates a confusion matrix, y_test is the real data, and yhat_test the predicted data
def saveWrongPredictions(y_test, yhat_test, filename):
    global permutation, X_test_documents
    #0 = bug, 1 = enhancement; => 0 = right classified; 1 = feature-bug = feature class as <bug>; -1 = bug-feature = bug classified as <feature>
    tmp = []
    for i in range(len(y_test)):
        if not y_test[i] == yhat_test[i]:
            label = "bug"
            if (yhat_test[i] == 1):
                label = "enhancement"
            tmp.append((label, X_test_documents[i]))
    
    array = ["."]*len(documentTexts)
    for i in range(len(permutation)):
        pos = permutation[i]
        if not y_test[i] == yhat_test[i]:
            array[pos] = "X"
        else:
            array[pos] = "_"
    
    f = open("antMap.txt", "w")
    f.write(str(array))
    f.close()

    writeToFile(filename, tmp)

#this is a stemmer
def stemmer(text):
    return [PorterStemmer().stem(token) for token in text]

#this is a lemmatizer
def lemmatizer(text):
    return [WordNetLemmatizer().lemmatize(token) for token in text]

#This method is used to convert the documents to actual numbers
#it returns the training data normalized to tfidf and the vectorized test data
def createFeatureVectors(X_train, X_test):
    global ngram, lowerCase, stripAccents, stopWords, numberToWordMapping, tidvectorizer
    #the vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
    vectorizer = TfidfVectorizer(tokenizer=None,\
        strip_accents=stripAccents, ngram_range=ngram,
        stop_words=stopWords,
        lowercase=lowerCase,
        min_df=2)
        
    X_train_vectorized = vectorizer.fit_transform(X_train)               #vectorisation
    X_test_vectorized = vectorizer.transform(X_test)
    numberToWordMapping = vectorizer.get_feature_names()
    tidvectorizer = vectorizer
    return X_train_vectorized, X_test_vectorized

#This method is used to train the given classifiers
def trainClassifiers(X_train_featureVector, X_test_featureVector, y_train, y_test, classifierObjectList):
    global listOfPropab
    predictions = []
    for classifierName,classifier in classifierObjectList:
        print(classifierName)
        classifier.fit(X_train_featureVector, y_train)
        predicted = classifier.predict(X_test_featureVector)
        scores.append(np.mean(predicted == y_test))
        print(metrics.classification_report(y_test, predicted,labels=[0,1], target_names=["bug","feature"]))
        if not classifierName == "sigmoidSVM":
            listOfPropab.append(classifier.predict_proba(X_test))
        predictions.append(predicted)
    return predictions
    

#This method is used for ensemble learning, to get a majority decission based on the weights
#it returns the decission of the classifiers
def classifierVoting(classifiers, x_train, x_test, y_train ,voteStrength=None):
    ensemble = VotingClassifier(estimators,weights=voteStrength, voting='hard')
    #fit the esemble model, after fitting the other ones
    ensemble.fit(x_train, y_train)#test our model on the test data
    plot_confusion_matrix(ensemble, x_test, y_test, normalize="all",display_labels=["bug","feature"])
    return ensemble.predict(x_test)


X_train, X_test, y_train, y_test = loadBugAndEnhancement()              #load the data
print("training on: {}% == {} documents\ntesting on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))

estimators=[('MultinomialNB', MultinomialNB()), \
    ('SGDClassifier', SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)),
    ('sigmoidSVM', SVC(kernel='sigmoid', gamma=1.0)),
    ('RandomForest', RandomForestClassifier(200, bootstrap=False)),
    #('BernoulliNB', BernoulliNB()),#the worst one
    ('LogisticRegression',LogisticRegression(solver='sag',random_state=100))
    ]


classifierPredictions = trainClassifiers(X_train, X_test, y_train, y_test, estimators)

#------------------------temporary Testing purpose---------------------------
#This is going to be deleted and jsut for testing purposes till 24.07

"""
classifier = 0
bug = np.array(list(map(lambda i: i[1], listOfPropab[classifier])))
issueNumber = np.argmin(bug, axis=0) #np.argmax(bug, axis=0)
issueNumberList = []
wordmap = {}

for i in range(10):
    temp = np.argmax(bug, axis=0)
    issueNumberList.append(temp)
    bug[temp] = 0
totalWords = []
for issueI in issueNumberList:
    className = "bug" if y_test[issueI] == 0 else "feature"
    classifiedAs = "bug" if classifierPredictions[classifier][issueI] == 0 else "feature"

    print("class: {}\t predicted: {}\tscore: {}".format(className, classifiedAs, listOfPropab[classifier][issueI][0]))
    wordlist = np.array(tidvectorizer.inverse_transform(X_test[issueI])).flatten()
    totalWords.append(wordlist)
    print(documentTexts[permutation[issueI]])
for wlist in totalWords:
    for word in wlist:
        if word in wordmap:
            wordmap[word] = wordmap[word]+1
        else:
            wordmap[word] = 1
print("-------------------------------")
for word in wordmap:
    if wordmap[word] > 1:
        print("{}\t{}".format(wordmap[word], word))

"""

#------------------------temporary Testing purpose---------------------------
#test your own documents here TODO own method and stuff for it
#testDocuments = ["bug", "enhancement", "feature", "add", "error",\
#    "is this a bug? If not, then I want to make a suggestion for a new feature", "howdy fellow coders, I have a problem at C:/users/..."]
#vecDocs = tidvectorizer.transform(testDocuments)
#for c_name, classifier in estimators:
#    for docIndex in range(len(testDocuments)):
#        className = "bug" if classifier.predict(vecDocs[docIndex]) == 0 else "feature"
#        if not c_name == "sigmoidSVM":
#            print("classifier: {}\tpredicted_class: <{}>\tpercentage: {}\t for: {}".format(c_name,className,\
#                classifier.predict_proba(vecDocs[docIndex]),
#                testDocuments[docIndex]))
#        else:
#            print("classifier: {}\tpredicted_class: <{}>\t for: {}".format(c_name,className,\
#                testDocuments[docIndex]))

#----------------------------------------------------------------------------


finalPrediction = classifierVoting(estimators, X_train, X_test, y_train, voteStrength=np.array(scores)/sum(scores))

print("ensemble-score:{}".format(np.mean(finalPrediction == y_test)))
print("trained on: {}% == {} documents\ntested on: {} documents".format(trainingPercentage, X_train.shape[0], X_test.shape[0]))
plt.show()
saveWrongPredictions(y_test, finalPrediction, "wrongClassified.json")

#bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)


"""
#commented out, because it's prediciton is not good it still has to many errors
k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)

knn = KNeighborsClassifier(n_neighbors=1)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train_vectorized, y_train)

grid_mean_scores = grid.cv_results_["mean_test_score"]
print(grid_mean_scores)
"""



""" #this is for tuning the parametes
parameters = {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],'penalty': ('l2', 'l1')}
gs_clf = GridSearchCV(svm, parameters, cv=5)
gs_clf = gs_clf.fit(X_train_vectorized, y_train)
for param_name in sorted(parameters.keys()):
    print("\t>{}:{}".format(param_name, gs_clf.best_params_[param_name]))
    print("\t{}".format(gs_clf.best_score_))

"""

