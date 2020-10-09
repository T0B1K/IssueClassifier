import numpy as np
import joblib

Vectorrizer = joblib.load('vectorizer.vz')

def getFromQueue(stringArray):
    return Vectorrizer.transform(stringArray)