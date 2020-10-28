import json
import joblib

rawConfig = open('../classifier/load_config.json', 'r')
loadConfig = json.load(rawConfig)
classifier_locations = loadConfig['classifier_locations']
rootFolder = loadConfig['classifierFolder']


def getClassifier(categories: list):
    """Method return classifier 

    Args:
        categories (list): categories  array of labels

    Returns:
        [type]: specific classifier
    """
    classifierPath = None
    for element in classifier_locations:
        if element['labels'] == categories:
            classifierPath = element['path']
    path: str = "{}/{}".format(rootFolder, classifierPath)
    assert classifierPath != None, "Categories: {}".format(categories)
    classifier = joblib.load(path)
    return classifier

def getVotingClassifier():
    """Voting Classifier

    Returns:
        [type]: classifier
    """
    classifierPath = loadConfig['voting']
    path: str = "{}/{}".format(rootFolder, classifierPath)
    classifier = joblib.load(path)
    return classifier

def getVectorizer():
    """Method returns Vectorizer

    Returns:
        [type]: vectorizer
    """
    vecpath = 'vectorizer.vz'
    vectorizer = joblib.load(vecpath)
    return vectorizer