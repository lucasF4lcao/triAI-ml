from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas import TriageRequest, TriageResponse
from shared.clinical import clinical_features
from shared.config import MODEL_PATH
from shared.decision import decide
from shared.features import VITAL_FEATURES
from shared.manchester import CRITICAL_COLOR, ESI_TO_COLOR, esi_to_color, has_critical_vitals
from shared.vocabulary import keyword_features

CONFIDENCE_DECIMALS = 4


def celsius_to_fahrenheit(celsius: float | None) -> float | None:
    return None if celsius is None else celsius * 9 / 5 + 32


class TriageClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        payload = joblib.load(model_path)
        self._model = payload["model"]
        self._features = payload["features"]
        self._weights = np.asarray(payload["decision_weights"])

    def classify(self, request: TriageRequest) -> TriageResponse:
        features = self._build_features(request)

        probabilities = self._model.predict_proba(features)[0]
        best = int(decide(probabilities[None, :], self._weights)[0])
        esi = int(self._model.classes_[best])

        critical = has_critical_vitals(
            oxygen_saturation=request.oxygen_saturation,
            systolic_pressure=request.systolic_pressure,
            heart_rate=request.heart_rate,
            respiratory_rate=request.respiratory_rate,
            temperature_celsius=request.temperature,
        )

        model_color = esi_to_color(esi, critical=False)
        color = esi_to_color(esi, critical=critical)

        return TriageResponse(
            predicted_color=color,
            confidence=round(float(probabilities[best]), CONFIDENCE_DECIMALS),
            probabilities={
                ESI_TO_COLOR[int(level)]: round(float(value), CONFIDENCE_DECIMALS)
                for level, value in zip(self._model.classes_, probabilities)
            },
            safety_rule_applied=color == CRITICAL_COLOR and model_color != CRITICAL_COLOR,
        )

    def _build_features(self, request: TriageRequest) -> pd.DataFrame:
        vitals = {
            **request.model_dump(include=set(VITAL_FEATURES)),
            "temperature": celsius_to_fahrenheit(request.temperature),
        }
        clinical = clinical_features(
            pd.DataFrame([vitals], columns=VITAL_FEATURES).apply(pd.to_numeric, errors="coerce")
        )
        row = {
            **clinical.iloc[0].to_dict(),
            "pain": request.pain,
            "arrived_by_ambulance": int(bool(request.ambulance)),
            **keyword_features(request.keywords or [], request.chief_complaint),
        }

        frame = pd.DataFrame([row], columns=self._features)
        return frame.apply(pd.to_numeric, errors="coerce")
