from pathlib import Path

from payment_risk_engine.data.transaction_loader import load_transactions
from payment_risk_engine.graph.transaction_graph import TransactionGraph

INPUT_PATH = Path("data/samples/transactions.jsonl")


def main() -> None:
    transactions = load_transactions(INPUT_PATH)
    transaction_graph = TransactionGraph.from_transactions(transactions)
    shared_beneficiaries = transaction_graph.shared_beneficiaries(minimum_sources=3)
    ranked_beneficiaries = sorted(shared_beneficiaries, key=lambda account_id: (transaction_graph.account_profile(account_id).incoming_accounts, transaction_graph.account_profile(account_id).incoming_amount), reverse=True)

    print(f"Transactions analysed: {len(transactions)}")
    print(f"Accounts in graph: {transaction_graph.number_of_accounts()}")
    print(f"Relationships in graph: {transaction_graph.number_of_relationships()}")
    print(f"Shared beneficiaries: {len(shared_beneficiaries)}")

    for beneficiary in ranked_beneficiaries[:10]:
        profile = transaction_graph.account_profile(beneficiary)
        print(f"Beneficiary {beneficiary}: {profile.incoming_accounts} source accounts, {profile.incoming_amount} EUR received")


if __name__ == "__main__":
    main()
