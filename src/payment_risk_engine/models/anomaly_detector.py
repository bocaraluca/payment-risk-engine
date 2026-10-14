from collections.abc import Sequence

from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self._model = IsolationForest(contamination=contamination, random_state=random_state)
        self._is_fitted = False

    def fit(self, feature_vectors: Sequence[Sequence[float]]) -> None:
        self._model.fit(feature_vectors)
        self._is_fitted = True

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("The model must be fitted before predicting anomaly scores.")

    def anomaly_score(self, feature_vector: Sequence[float]) -> float:
        self._ensure_fitted()

        normality_score = self._model.decision_function([feature_vector])[0]
        anomaly_score = -normality_score

        return float(anomaly_score)

    def is_anomaly(self, feature_vector: Sequence[float]) -> bool:
        self._ensure_fitted()

        prediction = self._model.predict([feature_vector])[0]
        return bool(prediction == -1)
