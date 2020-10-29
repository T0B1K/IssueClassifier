import json
import joblib

import configuration

config = configuration.Configuration()

classifierLocations = config.getValueFromConfig("classifierLocations")
rootFolder = config.getValueFromConfig("classifierFolder")


def getClassifier (categories:list):
    """Method return classifier

    Args:
        categories (list): Array of labels.

    Returns:
        [type]: The corresponding classifier.
    """
    if not categories:
        raise("There are no categories provided") 
    classifierPath = None
    for element in classifierLocations:
        if element['labels'] == categories:
            classifierPath =  element['path']
    path: str = "{}/{}".format(rootFolder,classifierPath)
    assert classifierPath != None ,"Categories: {}".format(categories)
    classifier = joblib.load(path)
    assert not classifier == None, "Classifier couldn't be loaded from {}".format(path)
    return classifier

def getVotingClassifier():
    """This method returns a voting classifier.

    Returns:
        [type]: The voting classifier if it could be loaded.
    """

    classifierPath = config.getValueFromConfig("trainingConstants voting")
    path:str = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    assert not classifier == None, "Classifier at {} couldn't be loaded".format(path)
    return classifier


def getVectorizer ():
    """Method returns Vectorizer.

    Returns:
        [type]: The tfidf vectorizer.
    """

    vecpath = config.getValueFromConfig("vectorizer path loadPath")
    vectorizer = joblib.load(vecpath)
    assert not vectorizer == None, "Vectorizer at {} couldn't be loaded".format(vecpath)
    return vectorizer


    