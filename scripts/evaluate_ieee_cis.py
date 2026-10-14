import argparse
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from payment_risk_engine.data.ieee_cis_loader import load_ieee_cis_dataset
from payment_risk_engine.models.supervised_fraud_detector import SupervisedFraudDetector

DEFAULT_DATASET_PATH = Path("data/raw/ieee_cis/train_transaction.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a supervised IEEE-CIS fraud detector.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for a quick experiment.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_ieee_cis_dataset(args.dataset, nrows=args.nrows)
    train_features, test_features, train_target, test_target = dataset.temporal_split()

    detector = SupervisedFraudDetector()
    detector.fit(train_features, train_target)
    probabilities = detector.predict_probability(test_features)
    predictions = (probabilities >= args.threshold).astype("int8")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_target, predictions, labels=[0, 1]).ravel()
    false_positive_rate = false_positive / (false_positive + true_negative)

    print(f"Rows: {len(dataset.features)}")
    print(f"Training rows: {len(train_features)}")
    print(f"Test rows: {len(test_features)}")
    print(f"Test fraud rate: {test_target.mean():.4%}")
    print(f"Precision: {precision_score(test_target, predictions, zero_division=0):.3f}")
    print(f"Recall: {recall_score(test_target, predictions, zero_division=0):.3f}")
    print(f"F1: {f1_score(test_target, predictions, zero_division=0):.3f}")
    print(f"PR-AUC: {average_precision_score(test_target, probabilities):.3f}")
    print(f"False-positive rate: {false_positive_rate:.3f}")
    print(f"Confusion matrix: TN={true_negative}, FP={false_positive}, FN={false_negative}, TP={true_positive}")


if __name__ == "__main__":
    main()
