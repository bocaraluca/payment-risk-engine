from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from payment_risk_engine.domain.transaction import Transaction, TransactionType
from payment_risk_engine.graph.transaction_graph import TransactionGraph


def create_transaction(sender_account_id, receiver_account_id, amount: str) -> Transaction:
    return Transaction(transaction_id=uuid4(), sender_account_id=sender_account_id, receiver_account_id=receiver_account_id, amount=Decimal(amount), currency="EUR", timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC), transaction_type=TransactionType.BANK_TRANSFER, device_id="device-001", country_code="RO")


def test_graph_identifies_shared_beneficiary() -> None:
    senders = [uuid4() for _ in range(3)]
    beneficiary = uuid4()
    transactions = [create_transaction(sender, beneficiary, "900") for sender in senders]

    transaction_graph = TransactionGraph.from_transactions(transactions)
    profile = transaction_graph.account_profile(beneficiary)

    assert transaction_graph.number_of_accounts() == 4
    assert transaction_graph.number_of_relationships() == 3
    assert beneficiary in transaction_graph.shared_beneficiaries(minimum_sources=3)
    assert profile.incoming_accounts == 3
    assert profile.incoming_transaction_count == 3
    assert profile.incoming_amount == Decimal("2700")
