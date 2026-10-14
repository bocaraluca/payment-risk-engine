import csv
from collections import Counter
from pathlib import Path

TRANSACTION_PATH = Path("data/raw/ieee_cis/train_transaction.csv")
IDENTITY_PATH = Path("data/raw/ieee_cis/train_identity.csv")


def count_transactions(file_path: Path) -> tuple[int, Counter[str]]:
    labels: Counter[str] = Counter()

    with file_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        row_count = 0
        for row in reader:
            row_count += 1
            labels[row["isFraud"]] += 1

    return row_count, labels


def count_rows(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8", newline="") as input_file:
        return sum(1 for _ in input_file) - 1


def main() -> None:
    transaction_count, labels = count_transactions(TRANSACTION_PATH)
    identity_count = count_rows(IDENTITY_PATH)
    fraud_count = labels["1"]
    fraud_rate = fraud_count / transaction_count

    print(f"Transaction rows: {transaction_count}")
    print(f"Identity rows: {identity_count}")
    print(f"Fraud rows: {fraud_count}")
    print(f"Legitimate rows: {labels['0']}")
    print(f"Fraud rate: {fraud_rate:.4%}")


if __name__ == "__main__":
    main()
