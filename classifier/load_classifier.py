import json
import joblib

import configuration

config = configuration.Configuration()

classifierLocations = config.getValueFromConfig("classifier_locations")
rootFolder = config.getValueFromConfig("classifierFolder")


def getClassifier (categories:list):
    """Method return classifier

    Args:
        categories (list): array of labels

    Returns:
        [type]: The corresponding classifier
    """
        
    classifierPath = None
    for element in classifierLocations:
        if element['labels'] == categories:
            classifierPath =  element['path']
    path: str = "{}/{}".format(rootFolder,classifierPath)
    assert classifierPath != None ,"Categories: {}".format(categories)
    classifier = joblib.load(path)
    return classifier

def getVotingClassifier():
    """This method returns a voting classifier

    Returns:
        [type]: The voting classifier if it could be loaded
    """

    classifierPath = config.getValueFromConfig("trainingConstants voting")
    path:str = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier


def getVectorizer ():
    """Method returns Vectorizer

    Returns:
        [type]: the vectorrizer
    """

    vecpath = config.getValueFromConfig("vectorrizer path loadPath")
    vectorizer = joblib.load(vecpath)
    return vectorizer


    