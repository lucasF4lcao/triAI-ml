import itertools

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from shared.decision import decide

CLASSES = [1, 2, 3, 4, 5]
WEIGHT_STEPS_GREEN = (1, 1.5, 2, 3, 4, 6)
WEIGHT_STEPS_BLUE = (1, 2, 4, 8, 16, 32)
MAX_URGENT_RECALL_LOSS = 0.02
MAX_ACCURACY_LOSS = 0.01


def _recalls(y_true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return recall_score(y_true, predicted, labels=CLASSES, average=None, zero_division=0)


def tune_decision_weights(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    unweighted = decide(probabilities, np.ones(5)) + 1
    baseline = _recalls(y_true, unweighted)
    baseline_accuracy = accuracy_score(y_true, unweighted)
    best_weights, best_score = np.ones(5), -1.0

    for green, blue in itertools.product(WEIGHT_STEPS_GREEN, WEIGHT_STEPS_BLUE):
        weights = np.array([1, 1, 1, green, blue], dtype=float)
        predicted = decide(probabilities, weights) + 1
        recalls = _recalls(y_true, predicted)

        if np.any(recalls[:2] < baseline[:2] - MAX_URGENT_RECALL_LOSS):
            continue
        if accuracy_score(y_true, predicted) < baseline_accuracy - MAX_ACCURACY_LOSS:
            continue

        score = f1_score(y_true, predicted, average="macro")
        if score > best_score:
            best_weights, best_score = weights, score

    return best_weights
