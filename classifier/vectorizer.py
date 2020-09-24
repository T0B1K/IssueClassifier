from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')

import joblib
import numpy as np
import fileManipulation

class Vectorrizer(fileManipulation.FileManipulation):
    def __init__(self, loadVec=True, saveVec=False, ngram=(1, 2), stripAccents=None, stopWords=None):
        super().__init__()
        self.Vecotrizer = self.prepareVectorizer(loadVec, saveVec, stripAccents, ngram, stopWords)

    #this is a stemmer
    def stemmer(self, text):
        return [PorterStemmer().stem(token) for token in text]

    #this is a lemmatizer
    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(token) for token in text]
    
    #This method is used to convert the documents to actual numbers
    #TODO maybe store
    #it returns the training data normalized to tfidf and the vectorized test data
    def createFeatureVectors(self, X_train_documents, X_test_documents):
        #the vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
        X_train_vectorized = self.Vecotrizer.transform(X_train_documents)    
        X_test_vectorized = self.Vecotrizer.transform(X_test_documents) #vectorisation
        return X_train_vectorized, X_test_vectorized
    
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
        
    def getSplitedDocs(self,sampleSize):
        length = len(self.labelClasses)
        docCount = round (sampleSize / length)
        docs = np.empty(0)
        for label in self.labelClasses:
            print("docs size: {} Byte".format(docs.itemsize))
            docs = np.append(self.getRandomDocs(label, docCount),docs)
        return docs
    