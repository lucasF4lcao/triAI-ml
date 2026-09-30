import joblib
import numpy as np
import pytest

from app.classifier import TriageClassifier
from app.schemas import TriageRequest

NORMAL = dict(
    heart_rate=72,
    systolic_pressure=118,
    diastolic_pressure=76,
    oxygen_saturation=98,
    temperature=36.5,
    respiratory_rate=14,
    pain=1,
)


class StubModel:
    classes_ = np.arange(1, 6)

    def predict_proba(self, frame):
        return np.tile([[0.05, 0.15, 0.45, 0.30, 0.05]], (len(frame), 1))


def classifier(tmp_path, weights) -> TriageClassifier:
    from shared.features import FEATURES

    path = tmp_path / "stub.pkl"
    joblib.dump({"model": StubModel(), "features": FEATURES, "decision_weights": weights}, path)
    return TriageClassifier(path)


def test_vitals_inside_the_normal_range_give_the_same_model_input(tmp_path):
    clf = classifier(tmp_path, [1, 1, 1, 1, 1])
    typical = clf._build_features(TriageRequest(**NORMAL, keywords=["toe-pain"]))
    other_normal = clf._build_features(
        TriageRequest(
            **{**NORMAL, "heart_rate": 100, "systolic_pressure": 140, "diastolic_pressure": 90, "oxygen_saturation": 95, "temperature": 37.7, "respiratory_rate": 20},
            keywords=["toe-pain"],
        )
    )

    assert typical.equals(other_normal)


def test_a_vital_outside_the_normal_range_changes_the_model_input(tmp_path):
    clf = classifier(tmp_path, [1, 1, 1, 1, 1])
    normal = clf._build_features(TriageRequest(**NORMAL))
    fast_heart = clf._build_features(TriageRequest(**{**NORMAL, "heart_rate": 130}))

    assert not normal.equals(fast_heart)
    assert fast_heart["hr_high"].iloc[0] == 30


def test_the_decision_weights_choose_the_color_and_the_confidence_is_the_real_probability(tmp_path):
    plain = classifier(tmp_path, [1, 1, 1, 1, 1]).classify(TriageRequest(**NORMAL))
    weighted = classifier(tmp_path, [1, 1, 1, 2, 1]).classify(TriageRequest(**NORMAL))

    assert (plain.predicted_color, plain.confidence) == ("Amarelo", 0.45)
    assert (weighted.predicted_color, weighted.confidence) == ("Verde", 0.30)


def test_sex_does_not_change_the_model_input(tmp_path):
    clf = classifier(tmp_path, [1, 1, 1, 1, 1])
    request = {**NORMAL, "keywords": ["back-pain"]}

    male = clf._build_features(TriageRequest(**request, sexo_masculino=True))
    female = clf._build_features(TriageRequest(**request, sexo_masculino=False))

    assert male.equals(female)
    assert "is_male" not in male.columns
