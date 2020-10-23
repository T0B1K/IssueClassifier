from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')

import joblib
import numpy
import file_manipulation
import load_classifier

"""
This class is used for creating / loading an Vectorizer, which is used to create feature vectors out of the documents
"""
class Vectorizer(file_manipulation.FileManipulation):

    """
    This is the constructor for the Vectorizer class
    Input:  loadVec (optional)      true is default - should the vectorizer be loaded
            saveVec (optional)      false is default- should the vectorizer be saved
            ngram   (optional)      (1,2) is default- means the word itself and the neighbor
            stripAccents (optional) should the accents be stripped 
            stopWords (optional)    should stopwords be removed
    """
    def __init__(self, labelClasses, loadVec=True, saveVec=False, ngram=(1, 2), stripAccents=None, stopWords=None):
        super().__init__()
        self.labelClasses = labelClasses
        self.Vecotrizer = self.prepareVectorizer(
            loadVec, saveVec, stripAccents, ngram, stopWords)

    """
    Description: This method is used for stemming tokens. I.e. cat should identify such strings as cats, catlike, and catty
    Input: List[String] of documents
    Output: List[String] of documents with stemmed tokens
    """

    def stemmer(self, text):
        return [PorterStemmer().stem(token) for token in text]

    """
    !Attention - currently not in use - just used for performance testing! TODO remove in final version 
    Description: This method is used for lemmatizing tokens. I.e.  "better" is mapped to "good" or "walking" to "walk"
    Input: List[String] of documents
    Output: List[String] of lemmatized documents
    """

    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(token) for token in text]

    """
    Description: This method is used for the vectorisation of the training and testing documents (creating tf-idf, ngram vectors out of the data)
    Input:  X_train_documents:  List[String] of unvectorized text documents
            X_test_documents:   List[String] of unvectorized text documents
    Output: X_train_vectorized: List[String] of tfidf vectorized text documents
            X_test_vectorized:  List[String] of tfidf vectorized text documents
    """

    def createFeatureVectors(self, X_train_documents, X_test_documents):
        # the vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
        X_train_vectorized = self.Vecotrizer.transform(X_train_documents)
        X_test_vectorized = self.Vecotrizer.transform(X_test_documents)  # vectorisation
        return X_train_vectorized, X_test_vectorized

    """
    Description: This method is used to load the vectorizer from an .vz file, if it exists, or to create a vectorizer
    Input:  loadVec: Boolean    whether the vectorizer should be loaded or not
            saveVec: Boolean    whether the vectorizer should be saved afterwards or not
    Output: an loaded or newly created TfidfVectorizer object
    """
    #TODO refactor
    def prepareVectorizer(self, loadVec, saveVec, stripAccents, ngram, stopWords):
        Vecotrizer = None
        if loadVec == True:
            return self.createNewVectorizer(loadVec, saveVec, stripAccents, ngram, stopWords)
        try:
            Vecotrizer = load_classifier.getVectorizer()
            return Vecotrizer
        except:
            return self.createNewVectorizer(loadVec, saveVec, stripAccents, ngram, stopWords)
    
    def createNewVectorizer(self, loadVec, saveVec, stripAccents, ngram, stopWords):
        train_Data = self.getSplitedDocs(
            file_manipulation.FileManipulation.values["sampleSize"])
        Vecotrizer = TfidfVectorizer(tokenizer=None,
            strip_accents=stripAccents, lowercase=None, ngram_range=ngram,
            stop_words=stopWords,
            min_df=2)
        Vecotrizer.fit_transform(train_Data)
        if saveVec == True:
            joblib.dump(Vecotrizer, '../vectorizer.vz', compress=9)
        return Vecotrizer


    """
    Description: This method is used for getting an equal amount of documents from each label class
    Input:  samplesize :int     how many documents should be returned
    Output: List[String]        the documents from the different label classes
    """

    def getSplitedDocs(self, sampleSize):
        length = len(self.labelClasses)
        docCount = round(sampleSize / length)
        docs = numpy.empty(0)
        for label in self.labelClasses:
            print("docs size: {} Byte".format(docs.itemsize))
            docs = numpy.append(self.getRandomDocs(label, docCount), docs)
        return docs
