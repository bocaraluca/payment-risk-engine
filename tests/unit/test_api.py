from fastapi.testclient import TestClient

from payment_risk_engine.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_transaction_endpoint() -> None:
    payload = {
        "transaction": {
            "transaction_id": "bdea0b7d-6f3d-5437-aa82-e1e1b3d3a48d",
            "sender_account_id": "2cf2c8b7-0bf5-5e43-992e-03c7c99f50b9",
            "receiver_account_id": "f6d92d65-483f-5af4-88bb-54037e09f9ec",
            "amount": "49.49",
            "currency": "EUR",
            "timestamp": "2026-01-01T06:49:00+00:00",
            "transaction_type": "card_payment",
            "device_id": "device-001",
            "country_code": "RO",
        },
        "transactions_last_hour": 1,
        "is_new_country": False,
    }

    response = client.post("/score-transaction", json=payload)

    assert response.status_code == 200
    assert response.json()["transaction_id"] == payload["transaction"]["transaction_id"]
    assert response.json()["decision"] in {"allow", "review"}
