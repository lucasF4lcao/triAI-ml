import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from shared.config import MODEL_DIR, MODEL_PATH
from shared.decision import decide
from shared.features import FEATURES, MONOTONIC_CST, TARGET
from shared.manchester import ESI_TO_COLOR
from shared.ordinal import OrdinalClassifier
from training.dataset import build_dataset
from training.tuning import tune_decision_weights

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.08,
    "random_state": RANDOM_STATE,
}


def evaluate(probabilities: np.ndarray, weights: np.ndarray, y_test: pd.Series) -> None:
    predicted = decide(probabilities, weights) + 1
    class_names = [f"ESI {esi} ({color})" for esi, color in ESI_TO_COLOR.items()]
    within_one_level = (np.abs(predicted - y_test.to_numpy()) <= 1).mean()

    logger.info("Pesos de decisão: %s", weights.tolist())
    logger.info("Acurácia: %.4f | acerto a até um nível: %.4f", accuracy_score(y_test, predicted), within_one_level)
    logger.info("\n%s", classification_report(y_test, predicted, target_names=class_names, zero_division=0))
    logger.info("Matriz de confusão:\n%s", confusion_matrix(y_test, predicted))
    logger.info(
        "Distribuição prevista x real: %s",
        {esi: (round(float((predicted == esi).mean()), 3), round(float((y_test == esi).mean()), 3)) for esi in ESI_TO_COLOR},
    )


def split(dataset: pd.DataFrame):
    x, y = dataset[FEATURES], dataset[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=VALIDATION_SIZE, random_state=1, stratify=y_train
    )
    return x_fit, x_val, x_test, y_fit, y_val, y_test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = build_dataset()
    logger.info("Registros: %s | Features: %s", f"{len(dataset):,}", len(FEATURES))
    x_fit, x_val, x_test, y_fit, y_val, y_test = split(dataset)

    model = OrdinalClassifier(MONOTONIC_CST, **MODEL_PARAMS).fit(x_fit, y_fit)
    weights = tune_decision_weights(model.predict_proba(x_val), y_val.to_numpy())
    evaluate(model.predict_proba(x_test), weights, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    payload = {"model": model, "features": FEATURES, "decision_weights": weights.tolist()}
    joblib.dump(payload, MODEL_PATH, compress=3)
    logger.info("Modelo salvo em %s", MODEL_PATH)


if __name__ == "__main__":
    main()
