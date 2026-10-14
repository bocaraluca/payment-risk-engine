import pandas as pd

from payment_risk_engine.data.ieee_cis_loader import IEEE_CIS_FEATURE_COLUMNS, IEEECISDataset
from payment_risk_engine.models.supervised_fraud_detector import SupervisedFraudDetector


def create_features(row_count: int = 30) -> pd.DataFrame:
    features = {column: [float(index) for index in range(row_count)] for column in IEEE_CIS_FEATURE_COLUMNS}
    features["ProductCD"] = ["W" if index % 2 else "C" for index in range(row_count)]
    features["card4"] = ["visa" if index % 2 else "mastercard" for index in range(row_count)]
    features["card6"] = ["debit" if index % 2 else "credit" for index in range(row_count)]
    features["P_emaildomain"] = ["gmail.com" if index % 2 else "yahoo.com" for index in range(row_count)]
    features["TransactionDT"] = list(range(row_count))
    return pd.DataFrame(features)


def test_ieee_cis_dataset_temporal_split() -> None:
    features = create_features()
    target = pd.Series([index % 2 for index in range(len(features))], dtype="int8")
    dataset = IEEECISDataset(features=features, target=target)

    train_features, test_features, train_target, test_target = dataset.temporal_split(test_fraction=0.2)

    assert len(train_features) == 24
    assert len(test_features) == 6
    assert train_features["TransactionDT"].max() < test_features["TransactionDT"].min()
    assert len(train_target) == 24
    assert len(test_target) == 6


def test_supervised_detector_returns_fraud_probabilities() -> None:
    features = create_features()
    target = pd.Series([index % 2 for index in range(len(features))], dtype="int8")
    detector = SupervisedFraudDetector()

    detector.fit(features, target)
    probabilities = detector.predict_probability(features.head(5))

    assert len(probabilities) == 5
    assert probabilities.between(0.0, 1.0).all()
