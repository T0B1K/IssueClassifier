import numpy as np
import joblib

vectorizer = joblib.load('./vectorizer/vectorizer.vz')

def get_from_queue(stringArray):
    return vectorizer.transform(stringArray)