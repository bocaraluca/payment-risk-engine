import json
from pathlib import Path

from score_sample_transactions import (
    INPUT_PATH,
    NORMAL_TRANSACTION_COUNT,
    score_transactions,
    train_detector,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from payment_risk_engine.data.transaction_loader import load_transactions

LABELS_PATH = Path("data/samples/transaction_labels.jsonl")


def load_labels(file_path: Path) -> dict[str, bool]:
    with file_path.open("r", encoding="utf-8") as input_file:
        return {record["transaction_id"]: bool(record["is_suspicious"]) for record in map(json.loads, input_file) if record}


def main() -> None:
    transactions = load_transactions(INPUT_PATH)
    labels = load_labels(LABELS_PATH)
    detector = train_detector(transactions)
    scored_transactions = score_transactions(transactions, detector)

    actual = [labels[str(score["transaction_id"])] for score in scored_transactions]
    predicted = [bool(score["is_anomaly"]) for score in scored_transactions]
    scores = [float(score["anomaly_score"]) for score in scored_transactions]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    false_positive_rate = false_positive / (false_positive + true_negative)

    print(f"Training samples: {NORMAL_TRANSACTION_COUNT}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.3f}")
    print(f"Recall: {recall_score(actual, predicted, zero_division=0):.3f}")
    print(f"F1: {f1_score(actual, predicted, zero_division=0):.3f}")
    print(f"PR-AUC: {average_precision_score(actual, scores):.3f}")
    print(f"False-positive rate: {false_positive_rate:.3f}")
    print(f"Confusion matrix: TN={true_negative}, FP={false_positive}, FN={false_negative}, TP={true_positive}")


if __name__ == "__main__":
    main()