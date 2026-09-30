import numpy as np
import pandas as pd
import pytest

from shared.decision import decide
from shared.features import FEATURES, MONOTONIC_CST
from shared.ordinal import OrdinalClassifier


@pytest.fixture(scope="module")
def model_and_frame():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(np.zeros((1500, len(FEATURES)), dtype=np.float32), columns=FEATURES)
    frame["hr_high"] = rng.uniform(0, 60, 1500)
    score = frame["hr_high"] / 15 + rng.normal(0, 0.8, 1500)
    target = np.clip(5 - np.floor(score).astype(int), 1, 5)

    return OrdinalClassifier(MONOTONIC_CST, max_iter=40, random_state=0).fit(frame, target), frame


def test_probabilities_are_valid(model_and_frame):
    model, frame = model_and_frame
    probabilities = model.predict_proba(frame.head(200))

    assert probabilities.shape == (200, 5)
    assert (probabilities >= 0).all()
    assert probabilities.sum(axis=1) == pytest.approx(1.0)


def test_a_worse_vital_never_makes_the_patient_less_urgent(model_and_frame):
    model, frame = model_and_frame
    grid = pd.DataFrame(np.zeros((40, len(FEATURES)), dtype=np.float32), columns=FEATURES)
    grid["hr_high"] = np.linspace(0, 60, 40)

    probabilities = model.predict_proba(grid)
    urgent_up_to_orange = probabilities[:, :2].sum(axis=1)

    assert (np.diff(urgent_up_to_orange) >= -1e-9).all()
    assert urgent_up_to_orange[-1] > urgent_up_to_orange[0]


def test_decision_weights_can_move_the_choice_to_a_rarer_class():
    probabilities = np.array([[0.05, 0.15, 0.45, 0.30, 0.05]])

    assert decide(probabilities, [1, 1, 1, 1, 1])[0] == 2
    assert decide(probabilities, [1, 1, 1, 2, 1])[0] == 3
