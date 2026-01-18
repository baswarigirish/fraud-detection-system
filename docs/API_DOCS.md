# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses API key authentication via the `X-API-Key` header.

**Demo API Key:** `demo-api-key-12345`

**Example:**
```bash
curl -H "X-API-Key: demo-api-key-12345" http://localhost:8000/predict
```

> **Note**: Authentication is optional in demo mode but recommended for production.

---

## Endpoints

### 1. Single Prediction

Predict fraud for a single transaction.

**Endpoint:** `POST /predict`

**Request Headers:**
```
Content-Type: application/json
X-API-Key: demo-api-key-12345 (optional in demo)
```

**Request Body:**
```json
{
  "Time": 12345,
  "V1": -0.5,
  "V2": 0.3,
  "V3": 1.2,
  "V4": -0.8,
  "V5": 0.1,
  "V6": -0.3,
  "V7": 0.5,
  "V8": -0.2,
  "V9": 0.7,
  "V10": -0.4,
  "V11": 0.2,
  "V12": 0.9,
  "V13": -0.6,
  "V14": 0.4,
  "V15": -0.1,
  "V16": 0.8,
  "V17": -0.3,
  "V18": 0.6,
  "V19": -0.5,
  "V20": 0.2,
  "V21": -0.7,
  "V22": 0.4,
  "V23": -0.2,
  "V24": 0.5,
  "V25": 0.3,
  "V26": -0.4,
  "V27": 0.1,
  "V28": -0.6,
  "Amount": 150.00
}
```

**Response (200 OK):**
```json
{
  "transaction_id": "txn_20250113_103045_abc123",
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "LOW",
  "explanation": {
    "V14": 0.15,
    "V4": -0.08,
    "Amount_Log": 0.05
  },
  "timestamp": "2025-01-13T10:30:45.123Z",
  "model_version": "fraud_ensemble_v1"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d @transaction.json
```

**Python Example:**
```python
import requests

transaction = {
    "Time": 12345,
    "V1": -0.5, "V2": 0.3, # ... V3-V28
    "Amount": 150.00
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction,
    headers={"X-API-Key": "demo-api-key-12345"}
)

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Risk: {result['risk_level']}")
```

---

### 2. Batch Prediction

Predict fraud for multiple transactions in a single request.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "transactions": [
    {
      "Time": 12345,
      "V1": -0.5,
      // ... V2-V28
      "Amount": 150.00
    },
    {
      "Time": 12346,
      "V1": 0.8,
      // ... V2-V28
      "Amount": 75.50
    }
    // ... up to 1000 transactions
  ]
}
```

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "transaction_id": "txn_20250113_103045_0001",
      "is_fraud": false,
      "fraud_probability": 0.23,
      "risk_level": "LOW",
      "explanation": {...},
      "timestamp": "2025-01-13T10:30:45.123Z",
      "model_version": "fraud_ensemble_v1"
    },
    {
      "transaction_id": "txn_20250113_103045_0002",
      "is_fraud": true,
      "fraud_probability": 0.87,
      "risk_level": "HIGH",
      "explanation": {...},
      "timestamp": "2025-01-13T10:30:45.124Z",
      "model_version": "fraud_ensemble_v1"
    }
  ],
  "batch_size": 2,
  "processing_time_ms": 156.78,
  "timestamp": "2025-01-13T10:30:45.125Z"
}
```

**Constraints:**
- Maximum batch size: 1000 transactions
- Minimum batch size: 1 transaction

---

### 3. Health Check

Check API health and status.

**Endpoint:** `GET /health`

**Response (200 OK - Healthy):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "fraud_ensemble_v1",
  "uptime_seconds": 3600.5,
  "redis_connected": true,
  "database_connected": true,
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

**Response (503 Service Unavailable - Unhealthy):**
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_version": "fraud_ensemble_v1",
  "uptime_seconds": 10.2,
  "redis_connected": false,
  "database_connected": true,
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

---

### 4. Metrics

Prometheus metrics for monitoring.

**Endpoint:** `GET /metrics`

**Response (200 OK):**
```
# TYPE fraud_api_requests_total counter
fraud_api_requests_total{method="POST",endpoint="/predict",status="200"} 1523

# TYPE fraud_api_request_duration_seconds histogram
fraud_api_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.05"} 1200
fraud_api_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.1"} 1450

# TYPE fraud_predictions_total counter
fraud_predictions_total{prediction="legitimate"} 1450
fraud_predictions_total{prediction="fraud"} 73
```

---

## Error Responses

### 400 Bad Request
Invalid input data.

```json
{
  "error": "Validation error",
  "detail": "Amount must be positive",
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

### 401 Unauthorized
Missing or invalid API key (if auth is enforced).

```json
{
  "error": "Unauthorized",
  "detail": "Missing or invalid API key",
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

### 422 Unprocessable Entity
Request validation failed.

```json
{
  "detail": [
    {
      "loc": ["body", "Amount"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

### 429 Too Many Requests
Rate limit exceeded.

```json
{
  "error": "Rate limit exceeded",
  "detail": "Maximum 100 requests per minute",
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

### 500 Internal Server Error
Server-side error.

```json
{
  "error": "Internal server error",
  "detail": "Model prediction failed",
  "timestamp": "2025-01-13T10:30:45.123Z"
}
```

---

## Response Fields

### PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | string | Unique transaction identifier |
| `is_fraud` | boolean | Binary fraud prediction |
| `fraud_probability` | float | Fraud probability (0-1) |
| `risk_level` | string | Risk category: LOW, MEDIUM, HIGH |
| `explanation` | object | Top 3 features (SHAP values) |
| `timestamp` | datetime | Prediction timestamp (ISO 8601) |
| `model_version` | string | Model version used |

### Risk Levels

| Risk Level | Probability Range | Recommended Action |
|------------|-------------------|-------------------|
| LOW | 0% - 30% | Allow transaction |
| MEDIUM | 30% - 70% | Additional verification |
| HIGH | 70% - 100% | Block & investigate |

---

## Rate Limiting

**Limits:**
- 100 requests per minute per IP address
- Applied to `/predict` and `/predict/batch` endpoints
- Excluded endpoints: `/health`, `/metrics`, `/`

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1673618400
```

---

## Performance

### Latency Targets

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| `/predict` | <50ms | <100ms | <200ms |
| `/predict/batch` (100 txns) | <800ms | <1200ms | <1500ms |

### Throughput

- Sustained: 150 req/s
- Burst: 500 req/s (with caching)

---

## SDKs & Examples

### Python Client

```python
import requests

class FraudDetectionClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
    
    def predict(self, transaction):
        response = requests.post(
            f"{self.base_url}/predict",
            json=transaction,
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, transactions):
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"transactions": transactions},
            headers={"X-API-Key": self.api_key}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = FraudDetectionClient(
    "http://localhost:8000",
    "demo-api-key-12345"
)

result = client.predict(transaction)
print(f"Fraud: {result['is_fraud']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const client = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'X-API-Key': 'demo-api-key-12345',
    'Content-Type': 'application/json'
  }
});

async function predictFraud(transaction) {
  const response = await client.post('/predict', transaction);
  return response.data;
}

// Usage
const result = await predictFraud(transaction);
console.log(`Fraud: ${result.is_fraud}`);
console.log(`Probability: ${result.fraud_probability}`);
```

---

## Interactive Documentation

Swagger UI available at: http://localhost:8000/docs

Features:
- Try out API endpoints
- View request/response schemas
- Test authentication
- See example requests
