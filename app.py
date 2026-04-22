from flask import Flask, request, jsonify
import joblib
import pandas as pd
import math
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
model = joblib.load(MODEL_PATH)

feature_cols = [
    "delivery_days",
    "delivery_vs_estimated",
    "price",
    "freight_value",
    "num_items",
    "num_sellers",
    "total_payment_value",
    "purchase_dayofweek",
    "log_price",
    "payment_type",
    "seller_state",
    "customer_state"
]

numeric_features = [
    "delivery_days",
    "delivery_vs_estimated",
    "price",
    "freight_value",
    "num_items",
    "num_sellers",
    "total_payment_value",
    "purchase_dayofweek",
    "log_price"
]

categorical_features = [
    "payment_type",
    "seller_state",
    "customer_state"
]

allowed_values = {
    "payment_type": [
        "boleto", "credit_card", "debit_card", "not_defined", "voucher"
    ],
    "seller_state": [
        "AC","AM","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA",
        "PB","PE","PI","PR","RJ","RN","RO","RS","SC","SE","SP"
    ],
    "customer_state": [
        "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
        "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"
    ]
}

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def validate_record(record):
    errors = {}

    missing = [col for col in feature_cols if col not in record]
    if missing:
        errors["missing_fields"] = missing
        return errors

    for col in numeric_features:
        if not is_number(record[col]):
            errors[col] = "must be a numeric value"

    if "price" in record and is_number(record["price"]) and record["price"] < 0:
        errors["price"] = "must be a positive number"

    if "freight_value" in record and is_number(record["freight_value"]) and record["freight_value"] < 0:
        errors["freight_value"] = "must be a positive number"

    if "total_payment_value" in record and is_number(record["total_payment_value"]) and record["total_payment_value"] < 0:
        errors["total_payment_value"] = "must be a positive number"

    if "num_items" in record and is_number(record["num_items"]) and record["num_items"] < 0:
        errors["num_items"] = "must be zero or greater"

    if "num_sellers" in record and is_number(record["num_sellers"]) and record["num_sellers"] < 0:
        errors["num_sellers"] = "must be zero or greater"

    if "purchase_dayofweek" in record and is_number(record["purchase_dayofweek"]):
        if record["purchase_dayofweek"] not in [0, 1, 2, 3, 4, 5, 6]:
            errors["purchase_dayofweek"] = "must be an integer from 0 to 6"

    for col in categorical_features:
        if record[col] not in allowed_values[col]:
            errors[col] = f"invalid value '{record[col]}'"

    if "price" in record and is_number(record["price"]):
        expected_log_price = math.log1p(record["price"])
        if "log_price" in record and is_number(record["log_price"]):
            if abs(record["log_price"] - expected_log_price) > 1e-6:
                errors["log_price"] = "must equal log1p(price)"

    return errors

def predict_from_records(records):
    df_input = pd.DataFrame(records)
    df_input = df_input[feature_cols]

    probabilities = model.predict_proba(df_input)[:, 1]
    predictions = model.predict(df_input)

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "prediction": int(pred),
            "probability": round(float(prob), 4),
            "label": "positive" if int(pred) == 1 else "negative"
        })
    return results

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "HW4 Customer Satisfaction API",
        "endpoints": ["/health", "/predict", "/predict/batch"]
    })

@app.route("/predict", methods=["POST"])
def predict_single():
    if not request.is_json:
        return jsonify({"error": "Invalid input", "details": "Request must be JSON"}), 400

    record = request.get_json()
    errors = validate_record(record)

    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    result = predict_from_records([record])[0]
    return jsonify(result)

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if not request.is_json:
        return jsonify({"error": "Invalid input", "details": "Request must be JSON"}), 400

    records = request.get_json()

    if not isinstance(records, list):
        return jsonify({"error": "Invalid input", "details": "Request body must be a JSON array"}), 400

    if len(records) == 0:
        return jsonify({"error": "Invalid input", "details": "Batch cannot be empty"}), 400

    if len(records) > 100:
        return jsonify({"error": "Invalid input", "details": "Batch size cannot exceed 100"}), 400

    batch_errors = {}
    for i, record in enumerate(records):
        errors = validate_record(record)
        if errors:
            batch_errors[f"record_{i}"] = errors

    if batch_errors:
        return jsonify({"error": "Invalid input", "details": batch_errors}), 400

    results = predict_from_records(records)
    return jsonify({"predictions": results, "count": len(results)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
