import json
import joblib

rawConfig = open ('../classifier/load_config.json','r') 
loadConfig = json.load(rawConfig)
classifier_locations = loadConfig['classifier_locations']
rootFolder = loadConfig['classifierFolder']


def getClassifier (categories):
    """
        Description: Method return classifier 
        Input:  categories  array of labels
        Output: classifier 
    """
    classifierPath = None
    for element in classifier_locations:
        if element['labels'] == categories:
            classifierPath =  element['path']
    path =  path = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier

def getVotingClassifier():
    """
        Description: Return Voting Classifier  
        Input:  categories  array of labels
        Output: classifier 
    """
    classifierPath = loadConfig['voting']
    path =  path = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier


def getVectorizer ():
    """
        Description: Method returns Vectorizer   
        Output: vectorizer
    """
    vecpath = 'vectorizer.vz'
    vectorizer = joblib.load(vecpath)
    return vectorizer


    