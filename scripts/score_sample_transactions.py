import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from payment_risk_engine.data.transaction_loader import load_transactions
from payment_risk_engine.domain.transaction import Transaction
from payment_risk_engine.features.transaction_features import extract_transaction_features
from payment_risk_engine.models.anomaly_detector import AnomalyDetector
from payment_risk_engine.services.risk_scoring import RiskScoringService

INPUT_PATH = Path("data/samples/transactions.jsonl")
OUTPUT_PATH = Path("data/processed/scored_transactions.jsonl")
NORMAL_TRANSACTION_COUNT = 200


def calculate_context(transaction: Transaction, history: dict[UUID, list[Transaction]], countries: dict[UUID, set[str]]) -> tuple[int, bool]:
    sender_history = history[transaction.sender_account_id]
    one_hour_ago = transaction.timestamp - timedelta(hours=1)
    transactions_last_hour = sum(previous.timestamp >= one_hour_ago for previous in sender_history)
    is_new_country = transaction.country_code not in countries[transaction.sender_account_id]
    return transactions_last_hour, is_new_country


def record_context(transaction: Transaction, history: dict[UUID, list[Transaction]], countries: dict[UUID, set[str]]) -> None:
    history[transaction.sender_account_id].append(transaction)
    countries[transaction.sender_account_id].add(transaction.country_code)


def train_detector(transactions: list[Transaction]) -> AnomalyDetector:
    history: dict[UUID, list[Transaction]] = defaultdict(list)
    countries: dict[UUID, set[str]] = defaultdict(set)
    training_vectors: list[tuple[float, ...]] = []

    for transaction in sorted(transactions[:NORMAL_TRANSACTION_COUNT], key=lambda item: item.timestamp):
        transactions_last_hour, is_new_country = calculate_context(transaction, history, countries)
        features = extract_transaction_features(transaction, transactions_last_hour, is_new_country)
        training_vectors.append(features.as_model_vector())
        record_context(transaction, history, countries)

    detector = AnomalyDetector(contamination=0.1)
    detector.fit(training_vectors)
    return detector


def score_transactions(transactions: list[Transaction], detector: AnomalyDetector) -> list[dict[str, object]]:
    history: dict[UUID, list[Transaction]] = defaultdict(list)
    countries: dict[UUID, set[str]] = defaultdict(set)
    service = RiskScoringService(detector)
    scored_transactions: list[dict[str, object]] = []

    for transaction in sorted(transactions, key=lambda item: item.timestamp):
        transactions_last_hour, is_new_country = calculate_context(transaction, history, countries)
        assessment = service.score_transaction(transaction, transactions_last_hour, is_new_country)
        scored_transactions.append({"transaction_id": str(assessment.transaction_id), "anomaly_score": assessment.anomaly_score, "is_anomaly": assessment.is_anomaly, "decision": assessment.decision.value})
        record_context(transaction, history, countries)

    return scored_transactions


def main() -> None:
    transactions = load_transactions(INPUT_PATH)
    detector = train_detector(transactions)
    scored_transactions = score_transactions(transactions, detector)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for scored_transaction in scored_transactions:
            output_file.write(json.dumps(scored_transaction) + "\n")

    anomaly_count = sum(item["is_anomaly"] for item in scored_transactions)
    print(f"Scored {len(scored_transactions)} transactions")
    print(f"Detected {anomaly_count} anomalies")
    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
