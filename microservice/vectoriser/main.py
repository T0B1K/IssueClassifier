from typing import List

import joblib
import numpy as np

vectoriser = joblib.load('./vectoriser/vectoriser.vz')


def get_from_queue(string_array: np.array) -> np.array:
    return vectoriser.transform(string_array)
