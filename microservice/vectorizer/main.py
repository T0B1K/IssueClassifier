import numpy as np
import joblib

vectorizer = joblib.load('./vectorizer/vectorizer.vz')

def get_from_queue(string_array):
    return vectorizer.transform(string_array)