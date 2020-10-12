import joblib
import numpy as np

folderPath = "./classifier/trainedClassifiers/"
trainedClassifiers = [joblib.load(folderPath + 'ensembleClassifier_enhancement-bug.joblib.pkl'),
    joblib.load(folderPath + 'ensembleClassifier_doku-api.joblib.pkl')]
classifierCategories = [("enhancement", "bug"), ("doku", "api")]

"""
    This method predicts the output given by a vectorrizer output and returns the labels zipped as list
    Input-representation of the numpy array: [[1,2,7,0,4], [0,2,67,3,1], ...]
    returns: [["bug", "api"], ["enhancement"], ...]
"""
def predict(vectorrizerOutput):
    finalLabels = []*vectorrizerOutput.shape[0]  #shape, bc the vectorrizer output is an numpy array
    for classifier, lbl in zip(trainedClassifiers, classifierCategories):
        prediction = classifier.predict(vectorrizerOutput)
        prediction = np.array(list(map ( lambda x : lbl[0] if x == 0 else lbl[1], prediction))) #now we have an array of lbls
        finalLabels.push(prediction)
    return list(zip(finalLabels))