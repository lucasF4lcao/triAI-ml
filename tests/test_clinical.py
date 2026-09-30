import numpy as np
import pandas as pd
import pytest

from shared.clinical import CLINICAL_FEATURES, clinical_features

NORMAL = dict(
    heart_rate=80,
    respiratory_rate=16,
    systolic_pressure=120,
    diastolic_pressure=75,
    temperature=98.2,
    oxygen_saturation=98,
)


def compute(**overrides) -> pd.Series:
    return clinical_features(pd.DataFrame([{**NORMAL, **overrides}])).iloc[0]


def test_every_vital_inside_its_normal_range_has_no_deviation():
    assert (compute() == 0).all()


@pytest.mark.parametrize(
    ("override", "feature", "expected"),
    [
        ({"heart_rate": 50}, "hr_low", 10),
        ({"heart_rate": 120}, "hr_high", 20),
        ({"respiratory_rate": 30}, "rr_high", 10),
        ({"systolic_pressure": 80}, "sbp_low", 20),
        ({"diastolic_pressure": 100}, "dbp_high", 10),
        ({"temperature": 101.0}, "temp_high", 1.0),
        ({"temperature": 95.0}, "temp_low", 1.8),
        ({"oxygen_saturation": 90}, "spo2_deficit", 5),
    ],
)
def test_deviation_is_the_distance_to_the_normal_range(override, feature, expected):
    assert compute(**override)[feature] == pytest.approx(expected, abs=1e-4)


def test_values_inside_the_range_anywhere_give_the_same_features():
    edge_low = compute(heart_rate=60, systolic_pressure=100, temperature=96.8, oxygen_saturation=95)
    edge_high = compute(heart_rate=100, systolic_pressure=140, temperature=100.0, oxygen_saturation=100)

    assert (edge_low == edge_high).all()
    assert (edge_low == compute()).all()


def test_a_missing_vital_stays_missing_so_the_model_can_treat_it():
    features = compute(heart_rate=np.nan)

    assert np.isnan(features["hr_low"]) and np.isnan(features["hr_high"])
    assert features["rr_low"] == 0


def test_the_feature_list_matches_what_is_computed():
    assert list(compute().index) == CLINICAL_FEATURES
