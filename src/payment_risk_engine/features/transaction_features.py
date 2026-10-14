from dataclasses import dataclass

from payment_risk_engine.domain.transaction import Transaction, TransactionType


@dataclass(frozen=True, slots=True)
class TransactionFeatures:
    amount: float
    hour: float
    is_night: float
    transactions_last_hour: float
    is_new_country: float
    transaction_type_code: float

    def as_model_vector(self) -> tuple[float, ...]:
        return (
            self.amount,
            self.hour,
            self.is_night,
            self.transactions_last_hour,
            self.is_new_country,
            self.transaction_type_code,
        )


_TRANSACTION_TYPE_CODES = {
    TransactionType.CARD_PAYMENT: 1.0,
    TransactionType.BANK_TRANSFER: 2.0,
    TransactionType.CASH_WITHDRAWAL: 3.0,
}


def extract_transaction_features(transaction: Transaction, transactions_last_hour: int, is_new_country: bool) -> TransactionFeatures:
    hour = transaction.timestamp.hour
    is_night = hour < 6 or hour >= 22

    return TransactionFeatures(amount=float(transaction.amount), hour=float(hour), is_night=float(is_night), transactions_last_hour=float(transactions_last_hour), is_new_country=float(is_new_country), transaction_type_code=_TRANSACTION_TYPE_CODES[transaction.transaction_type])
