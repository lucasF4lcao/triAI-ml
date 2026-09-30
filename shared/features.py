from shared.clinical import CLINICAL_FEATURES
from shared.vocabulary import KEYWORD_FEATURES

TARGET = "acuity"

VITAL_FEATURES = [
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "systolic_pressure",
    "diastolic_pressure",
]

BASE_FEATURES = [*CLINICAL_FEATURES, "pain", "arrived_by_ambulance"]

FEATURES = [*BASE_FEATURES, *KEYWORD_FEATURES]

MONOTONIC_INCREASING = {*CLINICAL_FEATURES, "pain", "arrived_by_ambulance"}
MONOTONIC_CST = [1 if feature in MONOTONIC_INCREASING else 0 for feature in FEATURES]
