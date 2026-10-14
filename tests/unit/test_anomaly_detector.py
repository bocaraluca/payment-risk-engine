import pytest

from payment_risk_engine.models.anomaly_detector import AnomalyDetector


def test_detector_identifies_anomalous_transaction() -> None:
    normal_vectors = [
        [
            100.0 + index,
            10.0 + index % 8,
            float(index % 2),
            1.0 + index % 3,
            float(index % 2),
            float(1 + index % 3),
        ]
        for index in range(100)
    ]
    anomalous_vector = [10000.0, 3.0, 1.0, 20.0, 1.0, 2.0]

    detector = AnomalyDetector(contamination=0.1)
    detector.fit(normal_vectors)

    assert detector.is_anomaly(anomalous_vector) is True
    assert detector.anomaly_score(anomalous_vector) > 0


def test_detector_requires_training_before_scoring() -> None:
    detector = AnomalyDetector()

    with pytest.raises(ValueError):
        detector.anomaly_score([100.0, 12.0, 0.0, 1.0, 0.0, 2.0])
