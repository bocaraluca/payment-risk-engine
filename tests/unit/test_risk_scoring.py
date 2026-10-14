from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from payment_risk_engine.domain.transaction import Transaction, TransactionType
from payment_risk_engine.models.anomaly_detector import AnomalyDetector
from payment_risk_engine.services.risk_scoring import RiskDecision, RiskScoringService


def create_transaction(amount: str, hour: int, country_code: str = "RO") -> Transaction:
    return Transaction(transaction_id=uuid4(), sender_account_id=uuid4(), receiver_account_id=uuid4(), amount=Decimal(amount), currency="EUR", timestamp=datetime(2026, 1, 1, hour, 0, tzinfo=UTC), transaction_type=TransactionType.BANK_TRANSFER, device_id="device-001", country_code=country_code)


def create_detector() -> AnomalyDetector:
    normal_vectors = [[100.0 + index, 10.0 + index % 8, float(index % 2), 1.0 + index % 3, float(index % 2), float(1 + index % 3)] for index in range(100)]
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(normal_vectors)
    return detector


def test_risk_scoring_marks_suspicious_transaction_for_review() -> None:
    service = RiskScoringService(detector=create_detector())
    transaction = create_transaction(amount="10000", hour=3, country_code="US")

    assessment = service.score_transaction(transaction=transaction, transactions_last_hour=20, is_new_country=True)

    assert assessment.transaction_id == transaction.transaction_id
    assert assessment.is_anomaly is True
    assert assessment.decision == RiskDecision.REVIEW
    assert assessment.anomaly_score > 0
    assert assessment.reasons == ("An unusual transaction pattern was detected",)


class AlwaysNormalDetector:
    def anomaly_score(self, feature_vector: tuple[float, ...]) -> float:
        return 0.0

    def is_anomaly(self, feature_vector: tuple[float, ...]) -> bool:
        return False


def test_risk_scoring_does_not_apply_hardcoded_rules() -> None:
    service = RiskScoringService(detector=AlwaysNormalDetector())
    transaction = create_transaction(amount="10000", hour=23, country_code="US")

    assessment = service.score_transaction(transaction=transaction, transactions_last_hour=100, is_new_country=True)

    assert assessment.is_anomaly is False
    assert assessment.decision == RiskDecision.ALLOW
    assert assessment.reasons == ()


def test_risk_scoring_returns_transaction_features() -> None:
    service = RiskScoringService(detector=create_detector())
    transaction = create_transaction(amount="250.75", hour=14)

    assessment = service.score_transaction(transaction=transaction, transactions_last_hour=1, is_new_country=False)

    assert assessment.features.amount == 250.75
    assert assessment.features.hour == 14.0
    assert assessment.decision in (RiskDecision.ALLOW, RiskDecision.REVIEW)
