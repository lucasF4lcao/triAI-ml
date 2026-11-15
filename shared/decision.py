import numpy as np


def decide(probabilities: np.ndarray, weights) -> np.ndarray:
    return np.argmax(probabilities * np.asarray(weights), axis=1)
