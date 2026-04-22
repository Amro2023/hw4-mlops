# HW4 — Customer Satisfaction Prediction API (MLOps)

## Overview
This project builds and deploys a machine learning model that predicts whether a customer review is positive based on order, delivery, and payment features from the Brazilian Olist e-commerce dataset.

The solution follows an end-to-end MLOps workflow:
- Model training and evaluation (HW2)
- Experiment tracking and model registry (HW4 Part 2)
- API development (Flask)
- Containerization (Docker)
- Cloud deployment (Render)

---

## Model Description
The final model is a **HistGradientBoostingClassifier** trained on engineered features including:

### Numeric Features
- delivery_days  
- delivery_vs_estimated  
- price  
- freight_value  
- num_items  
- num_sellers  
- total_payment_value  
- purchase_dayofweek  
- log_price  

### Categorical Features
- payment_type  
- seller_state  
- customer_state  

The model outputs:
- binary prediction (0 = negative, 1 = positive)
- probability score
- human-readable label

---

## Live API

Base URL:  
https://hw4-mlops-yq35.onrender.com  

Health endpoint:  
https://hw4-mlops-yq35.onrender.com/health  

---

## API Endpoints

### 1. Health Check
**GET /health**

Response:
```json
{
  "status": "healthy",
  "model": "loaded"
}
```

---

### 2. Single Prediction
**POST /predict**

Example request:
```json
{
  "delivery_days": 10,
  "delivery_vs_estimated": 2,
  "price": 150.0,
  "freight_value": 20.0,
  "num_items": 1,
  "num_sellers": 1,
  "total_payment_value": 170.0,
  "purchase_dayofweek": 2,
  "log_price": 5.017279836814924,
  "payment_type": "credit_card",
  "seller_state": "SP",
  "customer_state": "RJ"
}
```

Example response:
```json
{
  "prediction": 1,
  "probability": 0.83,
  "label": "positive"
}
```

---

### 3. Batch Prediction
**POST /predict/batch**

Accepts a list of records and returns predictions for each.

---

## Local Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API locally
```bash
PORT=8000 python3 app.py
```

### Run tests
```bash
python3 test_api.py
```

---

## Testing

The API was validated using a test script that includes:
- Health endpoint test  
- Valid single prediction test  
- Valid batch prediction test  
- Missing field validation (400 error)  
- Invalid type validation (400 error)  

All tests passed successfully.

---

## Docker

### Build image
```bash
docker build -t hw4-api .
```

### Run container
```bash
docker run -p 8000:8000 hw4-api
```

The application runs using Gunicorn for production readiness.

---

## Deployment

The API is deployed on Render using Docker.

Key deployment considerations:
- Environment-based port binding (`PORT`)
- Dependency version pinning (scikit-learn 1.6.1)
- Production server (Gunicorn)

The deployed service was tested using live API requests and returned valid predictions.

---

## Repository Structure

```text
hw4-mlops/
├── app.py
├── test_api.py
├── requirements.txt
├── Dockerfile
├── README.md
└── model/
    └── model.pkl
```

---

## Notes

- The model pipeline is serialized and loaded directly in the API  
- Input validation ensures robust handling of incorrect requests  
- Version consistency is maintained to avoid model loading errors  
- The API is production-ready and cloud-deployed  
