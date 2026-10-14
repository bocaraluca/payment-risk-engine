import json
from decimal import Decimal

import pytest

from payment_risk_engine.data.transaction_loader import load_transactions


def test_load_transactions_reads_jsonl_file(tmp_path) -> None:
    transaction_data = {
        "transaction_id": "bdea0b7d-6f3d-5437-aa82-e1e1b3d3a48d",
        "sender_account_id": "2cf2c8b7-0bf5-5e43-992e-03c7c99f50b9",
        "receiver_account_id": "f6d92d65-483f-5af4-88bb-54037e09f9ec",
        "amount": 49.49,
        "currency": "EUR",
        "timestamp": "2026-01-01T06:49:00+00:00",
        "transaction_type": "card_payment",
        "device_id": "device-001",
        "country_code": "RO",
    }
    input_path = tmp_path / "transactions.jsonl"
    input_path.write_text(json.dumps(transaction_data) + "\n", encoding="utf-8")

    transactions = load_transactions(input_path)

    assert len(transactions) == 1
    assert transactions[0].amount == Decimal("49.49")
    assert transactions[0].currency == "EUR"


def test_load_transactions_rejects_invalid_data(tmp_path) -> None:
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text("{\"amount\": -10}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_transactions(input_path)
