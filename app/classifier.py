from pathlib import Path

import joblib
import pandas as pd

from app.schemas import TriageRequest, TriageResponse
from shared.config import MODEL_PATH
from shared.features import VITAL_FEATURES
from shared.manchester import esi_to_color, has_critical_vitals
from shared.vocabulary import keyword_features

CONFIDENCE_DECIMALS = 4


def celsius_to_fahrenheit(celsius: float | None) -> float | None:
    return None if celsius is None else celsius * 9 / 5 + 32


class TriageClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        payload = joblib.load(model_path)
        self._model = payload["model"]
        self._features = payload["features"]

    def classify(self, request: TriageRequest) -> TriageResponse:
        features = self._build_features(request)

        probabilities = self._model.predict_proba(features)[0]
        best = probabilities.argmax()
        esi = int(self._model.classes_[best])

        critical = has_critical_vitals(
            oxygen_saturation=request.oxygen_saturation,
            systolic_pressure=request.systolic_pressure,
            heart_rate=request.heart_rate,
            respiratory_rate=request.respiratory_rate,
            temperature_celsius=request.temperature,
        )

        return TriageResponse(
            predicted_color=esi_to_color(esi, critical=critical),
            confidence=round(float(probabilities[best]), CONFIDENCE_DECIMALS),
        )

    def _build_features(self, request: TriageRequest) -> pd.DataFrame:
        row = {
            **request.model_dump(include={*VITAL_FEATURES, "pain"}),
            "temperature": celsius_to_fahrenheit(request.temperature),
            "arrived_by_ambulance": int(bool(request.ambulance)),
            "is_male": int(bool(request.is_male)),
            **keyword_features(request.keywords or [], request.chief_complaint),
        }

        frame = pd.DataFrame([row], columns=self._features)
        return frame.apply(pd.to_numeric, errors="coerce")
