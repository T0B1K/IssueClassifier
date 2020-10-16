import os

import joblib
import numpy as np

VECTORISER_PATH = os.environ["VECTORISER_PATH"] or './vectoriser/vectoriser.vz'

vectoriser = joblib.load(VECTORISER_PATH)


def vectorise_issues(string_array: np.array) -> np.array:
    return vectoriser.transform(string_array)
