import requests
import math

BASE_URL = "http://127.0.0.1:8000"

valid_record = {
    "delivery_days": 10,
    "delivery_vs_estimated": 2,
    "price": 150.0,
    "freight_value": 20.0,
    "num_items": 1,
    "num_sellers": 1,
    "total_payment_value": 170.0,
    "purchase_dayofweek": 2,
    "log_price": math.log1p(150.0),
    "payment_type": "credit_card",
    "seller_state": "SP",
    "customer_state": "RJ"
}

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model" in data
    print("Test 1 passed: /health")

def test_single_prediction():
    r = requests.post(f"{BASE_URL}/predict", json=valid_record)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    print("Test 2 passed: /predict valid single request")

def test_batch_prediction():
    batch = [valid_record.copy() for _ in range(5)]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 5
    print("Test 3 passed: /predict/batch valid batch request")

def test_missing_field():
    bad_record = valid_record.copy()
    del bad_record["price"]
    r = requests.post(f"{BASE_URL}/predict", json=bad_record)
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    print("Test 4 passed: missing required field returns 400")

def test_invalid_type():
    bad_record = valid_record.copy()
    bad_record["price"] = "one hundred"
    r = requests.post(f"{BASE_URL}/predict", json=bad_record)
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    print("Test 5 passed: invalid type returns 400")

if __name__ == "__main__":
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_missing_field()
    test_invalid_type()
    print("\nAll 5 tests passed successfully.")
