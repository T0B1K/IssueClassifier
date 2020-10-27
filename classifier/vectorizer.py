from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')
import logging

import joblib
import numpy
import file_manipulation
import load_classifier

"""
This class is used for creating / loading an Vectorizer, which is used to create feature vectors out of the documents
"""


class Vectorizer(file_manipulation.FileManipulation):

    def __init__(self, ngram: tuple = (1, 2), stripAccents=None, stopWords=None):
        """
        This is the constructor for the Vectorrizer class
        Input:  loadVec (optional)      true is default - should the vectorrizer be loaded
                saveVec (optional)      false is default- should the vectorrizer be saved
                ngram   (optional)      (1,2) is default- means the word itself and the neighbor
                stripAccents (optional) should the accents be stripped 
                stopWords (optional)    should stopwords be removed
        """
        super().__init__()
        self.Vecotrizer = self.prepareVectorizer(
            stripAccents, ngram, stopWords)

    def stemmer(self, text: list) -> list:
        """
        Description: This method is used for stemming tokens. I.e. cat should identify such strings as cats, catlike, and catty
        Input: List[String] of documents
        Output: List[String] of documents with stemmed tokens
        """
        return [PorterStemmer().stem(token) for token in text]

    def lemmatizer(self, text: list) -> list:
        """
        !Attention - currently not in use - just used for performance testing! TODO remove in final version 
        Description: This method is used for lemmatizing tokens. I.e.  "better" is mapped to "good" or "walking" to "walk"
        Input: List[String] of documents
        Output: List[String] of lemmatized documents
        """
        return [WordNetLemmatizer().lemmatize(token) for token in text]

    def createFeatureVectors(self, X_train_documents: numpy.ndarray, X_test_documents: numpy.ndarray) -> tuple:
        """
        Description: This method is used for the vectorisation of the training and testing documents (creating tf-idf, ngram vectors out of the data)
        Input:  X_train_documents:  List[String] of unvectorized text documents
                X_test_documents:   List[String] of unvectorized text documents
        Output: X_train_vectorized: List[String] of tfidf vectorized text documents
                X_test_vectorized:  List[String] of tfidf vectorized text documents
        """
        # the vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
        X_train_vectorized: scipy.sparse.csr.csr_matrix = self.Vecotrizer.transform(
            X_train_documents)
        X_test_vectorized: scipy.sparse.csr.csr_matrix = self.Vecotrizer.transform(
            X_test_documents)  # vectorisation
        return X_train_vectorized, X_test_vectorized

    # TODO refactor
    def prepareVectorizer(self, stripAccents, ngram, stopWords) -> TfidfVectorizer:
        """
        Description: This method is used to load the vectorrizer from an .vz file, if it exists, or to create a vectorrizer
        Input:  loadVec: Boolean    whether the vectorrizer should be loaded or not
                saveVec: Boolean    whether the vectorrizer should be saved afterwards or not
        Output: an loaded or newly created TfidfVectorizer object
        """
        loadVec:bool = file_manipulation.FileManipulation.values["vectorrizer"]["loadVectorizer"]
        if loadVec == True:
            return self.createNewVectorizer(stripAccents, ngram, stopWords)
        try:
            return load_classifier.getVectorizer() 
        except:
            return self.createNewVectorizer(stripAccents, ngram, stopWords)

    # TODO pls add comment
    def createNewVectorizer(self, stripAccents, ngram: tuple, stopWords) -> TfidfVectorizer:
        saveVec:bool = file_manipulation.FileManipulation.values["vectorrizer"]["saveVectorrizer"]

        train_Data:numpy.ndarray = self.getSplitedDocs(
            file_manipulation.FileManipulation.values["trainingConstants"]["sampleSize"])
        Vecotrizer = TfidfVectorizer(tokenizer=None,
                                     strip_accents=stripAccents, lowercase=None, ngram_range=ngram,
                                     stop_words=stopWords,
                                     min_df=2)
        Vecotrizer.fit_transform(train_Data)
        if saveVec == True:
            joblib.dump(
                Vecotrizer, file_manipulation.FileManipulation.values["vectorrizer"]["path"]["saveTo"], compress=9)
        return Vecotrizer

    def getSplitedDocs(self, sampleSize:int) -> numpy.ndarray:
        """
        Description: This method is used for getting an equal amount of documents from each label class
        Input:  samplesize :int     how many documents should be returned
        Output: List[String]        the documents from the different label classes
        """
        labelClasses:list = file_manipulation.FileManipulation.values["labelClasses"]
        length:int = len(labelClasses)
        docCount:float = round(sampleSize / length)
        docs:numpy.ndarray = numpy.empty(0)
        for label in labelClasses:
            logging.debug("docs size: {} Byte".format(docs.itemsize))
            docs = numpy.append(self.getRandomDocs(label, docCount), docs)
        return docs
