import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class OrdinalClassifier:
    classes_ = np.arange(1, 6)

    def __init__(self, monotonic_cst, **params):
        self.monotonic_cst = monotonic_cst
        self.params = params

    def fit(self, x, y):
        self.models_ = [
            HistGradientBoostingClassifier(monotonic_cst=self.monotonic_cst, **self.params).fit(
                x, (np.asarray(y) <= level).astype(int)
            )
            for level in self.classes_[:-1]
        ]
        return self

    def predict_proba(self, x) -> np.ndarray:
        cumulative = np.column_stack([model.predict_proba(x)[:, 1] for model in self.models_])
        cumulative = np.maximum.accumulate(cumulative, axis=1)
        edges = np.column_stack([np.zeros(len(cumulative)), cumulative, np.ones(len(cumulative))])
        return np.diff(edges, axis=1)
