from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from pydantic import BaseModel, Field

from payment_risk_engine.data.transaction_loader import load_transactions
from payment_risk_engine.domain.transaction import Transaction
from payment_risk_engine.features.transaction_features import extract_transaction_features
from payment_risk_engine.models.anomaly_detector import AnomalyDetector
from payment_risk_engine.services.risk_scoring import RiskDecision, RiskScoringService


class ScoreTransactionRequest(BaseModel):
    transaction: Transaction
    transactions_last_hour: int = Field(default=0, ge=0)
    is_new_country: bool = False


class ScoreTransactionResponse(BaseModel):
    transaction_id: UUID
    anomaly_score: float
    is_anomaly: bool
    decision: RiskDecision
    reasons: list[str]


def create_demo_service() -> RiskScoringService:
    project_root = Path(__file__).resolve().parents[3]
    sample_path = project_root / "data" / "samples" / "transactions.jsonl"
    transactions = load_transactions(sample_path)
    normal_transactions = transactions[:200]
    history: dict[UUID, list[Transaction]] = defaultdict(list)
    countries: dict[UUID, set[str]] = defaultdict(set)
    training_vectors: list[tuple[float, ...]] = []

    for transaction in sorted(normal_transactions, key=lambda item: item.timestamp):
        sender_history = history[transaction.sender_account_id]
        one_hour_ago = transaction.timestamp - timedelta(hours=1)
        transactions_last_hour = sum(previous.timestamp >= one_hour_ago for previous in sender_history)
        is_new_country = transaction.country_code not in countries[transaction.sender_account_id]
        features = extract_transaction_features(transaction, transactions_last_hour, is_new_country)
        training_vectors.append(features.as_model_vector())
        sender_history.append(transaction)
        countries[transaction.sender_account_id].add(transaction.country_code)

    detector = AnomalyDetector(contamination=0.1)
    detector.fit(training_vectors)
    return RiskScoringService(detector)


def create_app(scoring_service: RiskScoringService | None = None) -> FastAPI:
    app = FastAPI(title="Payment Risk Engine", version="0.1.0")
    service = scoring_service or create_demo_service()

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score-transaction", response_model=ScoreTransactionResponse)
    def score_transaction(request: ScoreTransactionRequest) -> ScoreTransactionResponse:
        assessment = service.score_transaction(request.transaction, request.transactions_last_hour, request.is_new_country)
        return ScoreTransactionResponse(transaction_id=assessment.transaction_id, anomaly_score=assessment.anomaly_score, is_anomaly=assessment.is_anomaly, decision=assessment.decision, reasons=list(assessment.reasons))

    return app


app = create_app()
