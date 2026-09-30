import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from shared.config import MODEL_DIR, MODEL_PATH
from shared.features import FEATURES, TARGET
from shared.manchester import ESI_TO_COLOR
from training.dataset import build_dataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.08,
    "random_state": RANDOM_STATE,
}


def evaluate(model: HistGradientBoostingClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    class_names = [f"ESI {esi} ({ESI_TO_COLOR[esi]})" for esi in model.classes_]
    within_one_level = (np.abs(predictions - y_test.to_numpy()) <= 1).mean()

    logger.info("Acurácia: %.4f | acerto a até um nível: %.4f", accuracy_score(y_test, predictions), within_one_level)
    logger.info("\n%s", classification_report(y_test, predictions, target_names=class_names))
    logger.info("Matriz de confusão:\n%s", confusion_matrix(y_test, predictions))


def cross_validate(x: pd.DataFrame, y: pd.Series) -> None:
    folds = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(HistGradientBoostingClassifier(**MODEL_PARAMS), x, y, cv=folds, scoring="accuracy")
    logger.info("Acurácia por fold: %s", [round(float(score), 4) for score in scores])
    logger.info("Acurácia média: %.4f (desvio %.4f)", scores.mean(), scores.std())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = build_dataset()
    x, y = dataset[FEATURES], dataset[TARGET]
    logger.info("Registros: %s | Features: %s", f"{len(x):,}", len(FEATURES))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(x_train, y_train)
    evaluate(model, x_test, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH, compress=3)
    logger.info("Modelo salvo em %s", MODEL_PATH)

    cross_validate(x, y)


if __name__ == "__main__":
    main()
