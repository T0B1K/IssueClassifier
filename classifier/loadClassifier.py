import json
import joblib

rawConfig = open ('classifier/loadConfig.json','r') 
loadConfig = json.load(rawConfig)
classifier_locations = loadConfig['classifier_locations']
rootFolder = loadConfig['classifierFolder']


def getClassifier (categories):
    classifierPath = None
    for element in classifier_locations:
        if element['labels'] == categories:
            classifierPath =  element['path']
    path =  path = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier

def getVortingClassifier():
    classifierPath = loadConfig['voting']
    path =  path = "{}/{}".format(rootFolder,classifierPath)
    classifier = joblib.load(path)
    return classifier


def getVectorizer ():
    vecpath = 'vectorizer.vz'
    vectorizer = joblib.load(vecpath)
    return vectorizer


    