from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from payment_risk_engine.data.ieee_cis_loader import IEEE_CIS_FEATURE_COLUMNS

CATEGORICAL_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain"]
NUMERIC_COLUMNS = [column for column in IEEE_CIS_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


class SupervisedFraudDetector:
    def __init__(self, random_state: int = 42) -> None:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        preprocessor = ColumnTransformer([("numeric", numeric_pipeline, NUMERIC_COLUMNS), ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)])
        classifier = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state)
        self._pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    def fit(self, features: pd.DataFrame, target: Sequence[int]) -> None:
        sample_weight = compute_sample_weight(class_weight="balanced", y=target)
        self._pipeline.fit(features, target, classifier__sample_weight=sample_weight)

    def predict_probability(self, features: pd.DataFrame) -> pd.Series:
        probabilities = self._pipeline.predict_proba(features)[:, 1]
        return pd.Series(probabilities, index=features.index, name="fraud_probability")

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        return (self.predict_probability(features) >= threshold).astype("int8")
