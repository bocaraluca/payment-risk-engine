from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TransactionType(StrEnum):
    CARD_PAYMENT = "card_payment"
    BANK_TRANSFER = "bank_transfer"
    CASH_WITHDRAWAL = "cash_withdrawal"


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: UUID
    sender_account_id: UUID
    receiver_account_id: UUID
    amount: Decimal = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    timestamp: datetime
    transaction_type: TransactionType
    device_id: str = Field(min_length=1)
    country_code: str = Field(min_length=2, max_length=2)
