from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from payment_risk_engine.domain.transaction import Transaction, TransactionType


def test_transaction_accepts_valid_data() -> None:
    transaction = Transaction(
        transaction_id=uuid4(),
        sender_account_id=uuid4(),
        receiver_account_id=uuid4(),
        amount=Decimal("125.50"),
        currency="EUR",
        timestamp=datetime.now(UTC),
        transaction_type=TransactionType.BANK_TRANSFER,
        device_id="device-001",
        country_code="RO",
    )

    assert transaction.amount == Decimal("125.50")
    assert transaction.currency == "EUR"


def test_transaction_rejects_negative_amount() -> None:
    with pytest.raises(ValidationError):
        Transaction(
            transaction_id=uuid4(),
            sender_account_id=uuid4(),
            receiver_account_id=uuid4(),
            amount=Decimal("-10"),
            currency="EUR",
            timestamp=datetime.now(UTC),
            transaction_type=TransactionType.BANK_TRANSFER,
            device_id="device-001",
            country_code="RO",
        )
