from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

import networkx as nx

from payment_risk_engine.domain.transaction import Transaction


@dataclass(frozen=True, slots=True)
class AccountGraphProfile:
    account_id: UUID
    incoming_accounts: int
    outgoing_accounts: int
    incoming_transaction_count: int
    outgoing_transaction_count: int
    incoming_amount: Decimal
    outgoing_amount: Decimal


class TransactionGraph:
    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    @classmethod
    def from_transactions(cls, transactions: list[Transaction]) -> "TransactionGraph":
        graph = nx.DiGraph()

        for transaction in transactions:
            sender = transaction.sender_account_id
            receiver = transaction.receiver_account_id
            graph.add_node(sender, node_type="account")
            graph.add_node(receiver, node_type="account")

            if graph.has_edge(sender, receiver):
                edge = graph[sender][receiver]
                edge["transaction_count"] += 1
                edge["total_amount"] += transaction.amount
            else:
                graph.add_edge(sender, receiver, transaction_count=1, total_amount=transaction.amount)

        return cls(graph)

    def account_profile(self, account_id: UUID) -> AccountGraphProfile:
        if account_id not in self._graph:
            return AccountGraphProfile(account_id, 0, 0, 0, 0, Decimal("0"), Decimal("0"))

        incoming_edges = list(self._graph.in_edges(account_id, data=True))
        outgoing_edges = list(self._graph.out_edges(account_id, data=True))

        return AccountGraphProfile(account_id=account_id, incoming_accounts=self._graph.in_degree(account_id), outgoing_accounts=self._graph.out_degree(account_id), incoming_transaction_count=sum(edge[2]["transaction_count"] for edge in incoming_edges), outgoing_transaction_count=sum(edge[2]["transaction_count"] for edge in outgoing_edges), incoming_amount=sum((edge[2]["total_amount"] for edge in incoming_edges), Decimal("0")), outgoing_amount=sum((edge[2]["total_amount"] for edge in outgoing_edges), Decimal("0")))

    def shared_beneficiaries(self, minimum_sources: int = 3) -> list[UUID]:
        return [account_id for account_id in self._graph if self._graph.in_degree(account_id) >= minimum_sources]

    def number_of_accounts(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_relationships(self) -> int:
        return self._graph.number_of_edges()
