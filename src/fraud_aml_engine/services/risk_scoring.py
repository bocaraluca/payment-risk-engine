from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from payment_risk_engine.domain.transaction import Transaction
from payment_risk_engine.features.transaction_features import (
    TransactionFeatures,
    extract_transaction_features,
)
from payment_risk_engine.models.anomaly_detector import AnomalyDetector


class RiskDecision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    transaction_id: UUID
    anomaly_score: float
    is_anomaly: bool
    decision: RiskDecision
    features: TransactionFeatures
    reasons: tuple[str, ...]


class RiskScoringService:
    def __init__(self, detector: AnomalyDetector) -> None:
        self._detector = detector

    def score_transaction(self, transaction: Transaction, transactions_last_hour: int, is_new_country: bool) -> RiskAssessment:
        features = extract_transaction_features(transaction=transaction, transactions_last_hour=transactions_last_hour, is_new_country=is_new_country)

        feature_vector = features.as_model_vector()
        anomaly_score = self._detector.anomaly_score(feature_vector)
        is_anomaly = self._detector.is_anomaly(feature_vector)
        reasons = ("An unusual transaction pattern was detected",) if is_anomaly else ()

        decision = RiskDecision.REVIEW if is_anomaly else RiskDecision.ALLOW

        return RiskAssessment(transaction_id=transaction.transaction_id, anomaly_score=anomaly_score, is_anomaly=is_anomaly, decision=decision, features=features, reasons=tuple(reasons))
