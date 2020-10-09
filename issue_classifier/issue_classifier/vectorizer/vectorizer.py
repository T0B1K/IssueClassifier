import joblib
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from issue_classifier.issue_data_methods.file_manipulation import FileManipulation


class Vectorizer(FileManipulation):

    def __init__(self, loadVec=True, saveVec=False, ngram=(1, 2), stripAccents=None, stopWords=None, label_classes=None):
        self.vectorzier = self.prepare_vectorizer(
            loadVec, saveVec, stripAccents, ngram, stopWords)
        self.label_classes = label_classes

    def stemmer(self, text):
        return [PorterStemmer().stem(token) for token in text]

    def lemmatizer(self, text):
        return [WordNetLemmatizer().lemmatize(token) for token in text]

    # This method is used to convert the documents to actual numbers
    # TODO maybe store
    # it returns the training data normalized to tfidf and the vectorized test data
    def create_feature_vectors(self, X_train_documents, X_test_documents):

        # The vectorizer is creating a vector out of the trainingsdata (bow) as well as removing the stopwords and emojis (non ascii) etc.
        X_train_vectorized = self.vectorzier.transform(X_train_documents)
        X_test_vectorized = self.vectorzier.transform(
            X_test_documents)  # vectorisation

        return X_train_vectorized, X_test_vectorized

    def prepare_vectorizer(self, load_vectorizer, save_vectorizer, strip_accents, ngram, stop_words):
        vectorizer = None

        if load_vectorizer == True:
            try:
                vectorizer = joblib.load('../vectorizer.vz', )

                return vectorizer
            except:
                print("Vectorizer could not be loaded")
                raise
        else:
            train_Data = self.get_splitted_documents(4000)
            vectorizer = TfidfVectorizer(tokenizer=None,
                                         strip_accents=strip_accents,
                                         lowercase=None,
                                         ngram_range=ngram,
                                         stop_words=stop_words,
                                         min_df=2)
            vectorizer.fit_transform(train_Data)
            if save_vectorizer == True:
                joblib.dump(vectorizer, '../vectorizer.vz', compress=9)

            return vectorizer

    def get_splitted_documents(self, sample_size):
        length = len(self.label_classes or "")
        docCount = round(sample_size / length)
        docs = np.empty(0)
        for label in self.label_classes:
            print("docs size: {} Byte".format(docs.itemsize))
            docs = np.append(self.get_random_documents(label, docCount), docs)
        return docs
