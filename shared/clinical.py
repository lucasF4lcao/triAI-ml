import pandas as pd

RANGES = {
    "heart_rate": (60.0, 100.0, "hr"),
    "respiratory_rate": (12.0, 20.0, "rr"),
    "systolic_pressure": (100.0, 140.0, "sbp"),
    "diastolic_pressure": (60.0, 90.0, "dbp"),
    "temperature": (96.8, 100.0, "temp"),
}
SPO2_FLOOR = 95.0

CLINICAL_FEATURES = [
    *(f"{prefix}_{side}" for _, _, prefix in RANGES.values() for side in ("low", "high")),
    "spo2_deficit",
]


def clinical_features(vitals: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=vitals.index)

    for column, (low, high, prefix) in RANGES.items():
        features[f"{prefix}_low"] = (low - vitals[column]).clip(lower=0)
        features[f"{prefix}_high"] = (vitals[column] - high).clip(lower=0)

    features["spo2_deficit"] = (SPO2_FLOOR - vitals["oxygen_saturation"]).clip(lower=0)
    return features.astype("float32")
