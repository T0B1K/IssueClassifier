from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')
import logging

import joblib
import numpy
import file_manipulation
import load_classifier
import configuration

config = configuration.Configuration()

class Vectorizer(file_manipulation.FileManipulation): 
    """This class is used for creating / loading an vectorizer, which is used to create feature vectors out of the documents.

    Args:
        file_manipulation (FileManipulation): This class is used to access data from documents as well as to save them later on.
    """

    def __init__(self, ngram: tuple = (1, 2), stripAccents=None, stopWords=None):
        """This is the constructor for the vectorizer class.

        Args:
            ngram (tuple, optional): How many neighbor words should be taken into consideration when creating an featurevector. Defaults to (1, 2).
            stripAccents (str, optional): Which accents should be stripped 'ASCII', 'UNICODE'. Defaults to None.
            stopWords (list, optional): Which stopwords should be removed i.e. {'english'}. Defaults to None.
        """
        
        super().__init__()
        self.Vecotrizer = self.prepareVectorizer(
            stripAccents, ngram, stopWords)

    def stemmer(self, text: list) -> list:
        """This method is used for stemming tokens. I.e. cat should identify such strings as cats, catlike, and catty.

        Args:
            text (list): A list of texts / documents.

        Returns:
            list: The same document list, but the tokens are now stemmed.
        """
        if not text:
            raise ("Parameter 'text' of type list is empty!")
        return [PorterStemmer().stem(token) for token in text]

    def lemmatizer(self, text: list) -> list:
        """!Attention - currently not in use - just used for performance testing! TODO remove in final version 
        This method can be used for lemmatizing tokens. I.e.  "better" is mapped to "good" or "walking" to "walk".

         Args:
            text (list): A list of texts / documents.

        Returns:
            list: The same document list, but the tokens are now stemmed.
        """
        if not text:
            raise ("Parameter 'text' of type list is empty!")
        return [WordNetLemmatizer().lemmatize(token) for token in text]

    def createFeatureVectors(self, X_train_documents: numpy.ndarray, X_test_documents: numpy.ndarray) -> tuple:
        """This method is used for the vectorization of the training and testing documents (creating tf-idf, ngram vectors out of the data)
        So it is basically creating a vector (number vector) out of the trainingsdata (string vector) and removes the stopwords, emojis, ... if selected

        Args:
            X_train_documents (numpy.ndarray): List of not vectorized text documents.
            X_test_documents (numpy.ndarray): List of not vectorized text documents.

        Returns:
            tuple:  X_train_vectorized (numpy.ndarray): List of tfidf weighted / vectorized test documents.
                    X_test_vectorized  (numpy.ndarray): List of tfidf weighted / vectorized train documents.
        """
        if not X_train_documents.size:
            raise("Parameter 'X_train_documents' of type numpy.ndarray is empty!")
        if not X_test_documents.size:
            raise("Parameter 'X_test_documents' of type numpy.ndarray is empty!")
        X_train_vectorized: scipy.sparse.csr.csr_matrix = self.Vecotrizer.transform(X_train_documents)
        X_test_vectorized: scipy.sparse.csr.csr_matrix = self.Vecotrizer.transform(X_test_documents)
        return X_train_vectorized, X_test_vectorized

    def prepareVectorizer(self, stripAccents, ngram, stopWords) -> TfidfVectorizer:
        """This method loads a vectorizer from an .vz file if this file exists. Otherwise it will create a new vectorizer

        Args:
            stripAccents (str, optional): Which accents should be stripped 'ASCII', 'UNICODE'. Defaults to None.
            ngram (tuple, optional): How many neighbor words should be taken into consideration when creating an featurevector. Defaults to (1, 2).
            stopWords (list, optional): Which stopwords should be removed i.e. {'english'}. Defaults to None.

        Returns:
            TfidfVectorizer: An loaded or newly created TfidfVectorizer object
        """
        loadVec:bool = config.getValueFromConfig("vectorizer loadVectorizer")
        if loadVec == False:
            return self.createNewVectorizer(stripAccents, ngram, stopWords)
        try:
            return load_classifier.getVectorizer() 
        except:
            return self.createNewVectorizer(stripAccents, ngram, stopWords)
    
    def createNewVectorizer(self, stripAccents, ngram: tuple, stopWords) -> TfidfVectorizer:
        """This method creates a new tfidf vectorizer and returns it

        Args:
            stripAccents (str, optional): Which accents should be stripped 'ASCII', 'UNICODE'. Defaults to None.
            ngram (tuple, optional): How many neighbor words should be taken into consideration when creating an featurevector. Defaults to (1, 2).
            stopWords (list, optional): Which stopwords should be removed i.e. {'english'}. Defaults to None.

        Returns:
            TfidfVectorizer: The newly created tfidf vectorizer object.
        """
        saveVec:bool = config.getValueFromConfig("vectorizer saveVectorizer")

        train_Data:numpy.ndarray = self.getSplitedDocs(config.getValueFromConfig("trainingConstants sampleSize"))
        Vecotrizer: TfidfVectorizer = TfidfVectorizer(tokenizer=None,
                                     strip_accents=stripAccents, lowercase=None, ngram_range=ngram,
                                     stop_words=stopWords,
                                     min_df=2)
        Vecotrizer.fit_transform(train_Data)
        if saveVec == True:
            joblib.dump(Vecotrizer, config.getValueFromConfig("vectorizer path saveTo"), compress=9)
        return Vecotrizer

    def getSplitedDocs(self, sampleSize:int) -> numpy.ndarray:
        """This method is used for getting an equal amount of documents from each label class
        by calculating the right amount and randomly selecting certain documents from each label class provided in the config

        Args:
            sampleSize (int): How many documents should be returned.

        Returns:
            numpy.ndarray: A list of {sampleSize} documents, with an equal amount of documents in each label class provided in the config.
        """   
        
        labelClasses:list = config.getValueFromConfig("labelClasses")
        length:int = len(labelClasses)
        docCount:float = round(sampleSize / length)
        docs:numpy.ndarray = numpy.empty(0)
        for label in labelClasses:
            logging.debug("docs size: {} Byte".format(docs.itemsize))
            docs = numpy.append(self.getRandomDocs(label, docCount), docs)
        return docs
