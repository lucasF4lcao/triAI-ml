import numpy as np
import pandas as pd

from shared.config import RAW_DATA_DIR
from shared.clinical import clinical_features
from shared.features import FEATURES, TARGET, VITAL_FEATURES
from shared.vocabulary import KEYWORD_FEATURES, keyword_matrix

TRIAGE_COLUMNS = [
    "stay_id", "temperature", "heartrate", "resprate", "o2sat",
    "sbp", "dbp", "pain", "chiefcomplaint", "acuity",
]
EDSTAYS_COLUMNS = ["stay_id", "arrival_transport"]

COLUMN_RENAMES = {
    "heartrate": "heart_rate",
    "resprate": "respiratory_rate",
    "o2sat": "oxygen_saturation",
    "sbp": "systolic_pressure",
    "dbp": "diastolic_pressure",
}

VITAL_LIMITS = {
    "temperature": (85.0, 110.0),
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (4.0, 80.0),
    "oxygen_saturation": (50.0, 100.0),
    "systolic_pressure": (40.0, 300.0),
    "diastolic_pressure": (20.0, 200.0),
}

PAIN_WORDS = {"critical": 10.0, "crit": 10.0, "mild": 3.0, "moderate": 5.0, "bad": 7.0, "alot": 7.0}
AMBULANCE_TRANSPORTS = ["AMBULANCE", "HELICOPTER"]


def load_raw_data() -> pd.DataFrame:
    triage = pd.read_csv(RAW_DATA_DIR / "triage.csv.gz", usecols=TRIAGE_COLUMNS)
    stays = pd.read_csv(RAW_DATA_DIR / "edstays.csv.gz", usecols=EDSTAYS_COLUMNS)

    merged = triage.merge(stays, on="stay_id", how="left", validate="one_to_one")
    merged = merged.rename(columns=COLUMN_RENAMES)
    merged = merged.dropna(subset=[TARGET])
    merged[TARGET] = merged[TARGET].astype(int)

    return merged.reset_index(drop=True)


def clean_pain(raw: pd.Series) -> pd.Series:
    text = raw.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    range_mean = text.str.extract(r"^(\d+\.?\d*)-(\d+\.?\d*)$").astype(float).mean(axis=1)

    pain = numeric.fillna(range_mean).fillna(text.map(PAIN_WORDS))
    return pain.where(pain.between(0, 10))


def clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    for column, (low, high) in VITAL_LIMITS.items():
        values = pd.to_numeric(df[column], errors="coerce")
        df[column] = values.where(values.between(low, high))
    return df


def keyword_columns(complaints: pd.Series) -> pd.DataFrame:
    matrix = keyword_matrix(complaints.fillna("")).toarray()
    return pd.DataFrame(matrix, columns=KEYWORD_FEATURES, index=complaints.index)


def build_dataset() -> pd.DataFrame:
    df = load_raw_data()
    df["pain"] = clean_pain(df["pain"])
    df = clean_vitals(df)
    df["arrived_by_ambulance"] = df["arrival_transport"].isin(AMBULANCE_TRANSPORTS).astype(np.uint8)

    clinical = clinical_features(df[VITAL_FEATURES])
    base = pd.concat([clinical, df[["pain", "arrived_by_ambulance"]]], axis=1).astype(np.float32)
    dataset = pd.concat([base, keyword_columns(df["chiefcomplaint"]), df[TARGET]], axis=1)
    return dataset[[*FEATURES, TARGET]]
