import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

COUNTRIES = ("RO", "DE", "FR", "ES", "IT")
TRANSACTION_TYPES = ("card_payment", "bank_transfer", "cash_withdrawal")


def make_id(prefix: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"payment-risk-engine:{prefix}:{index}"))


def create_normal_transaction(index: int, start_time: datetime, rng: random.Random) -> dict[str, object]:
    sender_index = rng.randrange(50)
    receiver_index = rng.randrange(50)
    timestamp = start_time + timedelta(minutes=rng.randrange(7 * 24 * 60))

    return {
        "transaction_id": make_id("normal-transaction", index),
        "sender_account_id": make_id("account", sender_index),
        "receiver_account_id": make_id("merchant", receiver_index),
        "amount": round(max(2.0, rng.lognormvariate(3.6, 0.55)), 2),
        "currency": "EUR",
        "timestamp": timestamp.isoformat(),
        "transaction_type": rng.choice(TRANSACTION_TYPES),
        "device_id": make_id("device", sender_index),
        "country_code": rng.choice(COUNTRIES),
    }


def create_smurfing_transaction(index: int, start_time: datetime) -> dict[str, object]:
    receiver_id = make_id("suspicious-beneficiary", 0)
    sender_id = make_id("smurfing-account", index % 10)
    timestamp = start_time + timedelta(minutes=index)

    return {
        "transaction_id": make_id("smurfing-transaction", index),
        "sender_account_id": sender_id,
        "receiver_account_id": receiver_id,
        "amount": round(900.0 + (index % 9) * 10.0, 2),
        "currency": "EUR",
        "timestamp": timestamp.isoformat(),
        "transaction_type": "bank_transfer",
        "device_id": make_id("suspicious-device", 0),
        "country_code": "US",
    }


def generate_transactions(normal_count: int, suspicious_count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    start_time = datetime(2026, 1, 1, tzinfo=UTC)
    normal_transactions = [create_normal_transaction(index, start_time, rng) for index in range(normal_count)]
    suspicious_transactions = [create_smurfing_transaction(index, start_time) for index in range(suspicious_count)]
    return normal_transactions + suspicious_transactions


def write_jsonl(transactions: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for transaction in transactions:
            output_file.write(json.dumps(transaction) + "\n")


def write_labels(transactions: list[dict[str, object]], normal_count: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, transaction in enumerate(transactions):
            label = {"transaction_id": transaction["transaction_id"], "is_suspicious": index >= normal_count}
            output_file.write(json.dumps(label) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic payment transactions.")
    parser.add_argument("--normal-count", type=int, default=200)
    parser.add_argument("--suspicious-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/samples/transactions.jsonl"))
    parser.add_argument("--labels-output", type=Path, default=Path("data/samples/transaction_labels.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions = generate_transactions(args.normal_count, args.suspicious_count, args.seed)
    write_jsonl(transactions, args.output)
    write_labels(transactions, args.normal_count, args.labels_output)
    print(f"Generated {len(transactions)} transactions in {args.output}")
    print(f"Generated labels in {args.labels_output}")


if __name__ == "__main__":
    main()
