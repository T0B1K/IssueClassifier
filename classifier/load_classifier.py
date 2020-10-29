import json
import joblib

import configuration

config = configuration.Configuration()

classifierLocations = config.getValueFromConfig("classifier_locations")
rootFolder = config.getValueFromConfig("classifierFolder")


def getClassifier (categories):
    """
        Description: Method return classifier 
        Input:  categories  array of labels
        Output: classifier 
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
    """
        Description: Return Voting Classifier  
        Input:  categories  array of labels
        Output: classifier 
    """
    classifierPath = config.getValueFromConfig("trainingConstants voting")
    path:str = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier


def getVectorizer ():
    """
        Description: Method returns Vectorizer   
        Output: vectorizer
    """
    vecpath = '../classifier/trained_classifiers/vectoriser.vz'
    vectorizer = joblib.load(vecpath)
    return vectorizer


    