import json
from pathlib import Path

from pydantic import ValidationError

from payment_risk_engine.domain.transaction import Transaction


def load_transactions(file_path: Path) -> list[Transaction]:
    transactions: list[Transaction] = []

    with file_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                transaction_data = json.loads(line)
                transaction = Transaction.model_validate(transaction_data)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(f"Invalid transaction on line {line_number}") from error

            transactions.append(transaction)

    return transactions
