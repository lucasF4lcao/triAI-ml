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

BASE_FEATURES = [*VITAL_FEATURES, "pain", "arrived_by_ambulance", "is_male"]

FEATURES = [*BASE_FEATURES, *KEYWORD_FEATURES]
