import numpy as np
import joblib

vectorizer = joblib.load('./vectorizer.vz')

def createFeatureVector(string_array):
    print("vectorrizing")
    return vectorizer.transform(string_array)