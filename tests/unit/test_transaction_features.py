from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from payment_risk_engine.domain.transaction import Transaction, TransactionType
from payment_risk_engine.features.transaction_features import extract_transaction_features


def test_extract_transaction_features() -> None:
    transaction = Transaction(
        transaction_id=uuid4(),
        sender_account_id=uuid4(),
        receiver_account_id=uuid4(),
        amount=Decimal("250.75"),
        currency="EUR",
        timestamp=datetime(2026, 1, 1, 23, 30, tzinfo=UTC),
        transaction_type=TransactionType.BANK_TRANSFER,
        device_id="device-001",
        country_code="RO",
    )

    features = extract_transaction_features(
        transaction=transaction,
        transactions_last_hour=5,
        is_new_country=True,
    )

    assert features.amount == 250.75
    assert features.hour == 23.0
    assert features.is_night == 1.0
    assert features.transactions_last_hour == 5.0
    assert features.is_new_country == 1.0
    assert features.transaction_type_code == 2.0
    assert len(features.as_model_vector()) == 6
