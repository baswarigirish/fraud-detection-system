# Quick Start Guide - Fraud Detection System

## Setup & Installation (5 minutes)

### Step 1: Download Dataset
```bash
# Install Kaggle API (if not already done)
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

# Download dataset
python scripts/download_data.py
```

### Step 2: Preprocess Data
```bash
python data/preprocessing.py
```

Expected output:
- `data/processed/train_latest.csv`
- `data/processed/val_latest.csv`
- `data/processed/test_latest.csv`
- `data/processed/scaler.pkl`

### Step 3: Train Models
```bash
python models/train.py
```

This will train all 3 models (takes ~10 minutes on CPU):
- Isolation Forest
- XGBoost
- Autoencoder

### Step 4: Evaluate Ensemble
```bash
python models/evaluate.py
```

Generates evaluation reports in `outputs/evaluation/`

---

## Running the API

### Option A: Docker (Recommended)
```bash
cd deployment
docker-compose up -d
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

### Option B: Local Development
```bash
# Start Redis (required for caching)
redis-server

# Start PostgreSQL (optional, for logging)
# Use Docker or local installation

# Start API
uvicorn api.main:app --reload --port 8000
```

---

## Testing the API

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-12345" \
  -d '{
    "Time": 12345,
    "V1": -0.5, "V2": 0.3, "V3": 1.2, "V4": -0.8,
    "V5": 0.1, "V6": -0.3, "V7": 0.5, "V8": -0.2,
    "V9": 0.7, "V10": -0.4, "V11": 0.2, "V12": 0.9,
    "V13": -0.6, "V14": 0.4, "V15": -0.1, "V16": 0.8,
    "V17": -0.3, "V18": 0.6, "V19": -0.5, "V20": 0.2,
    "V21": -0.7, "V22": 0.4, "V23": -0.2, "V24": 0.5,
    "V25": 0.3, "V26": -0.4, "V27": 0.1, "V28": -0.6,
    "Amount": 150.00
  }'
```

### Using Python
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

print(response.json())
```

---

## Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html  # Mac
# or
start htmlcov/index.html  # Windows
```

---

## Load Testing
```bash
# Test with 1000 requests
python scripts/simulate_traffic.py --requests 1000

# Test with generated data
python scripts/simulate_traffic.py --requests 500 --generated
```

---

## Complete Pipeline (End-to-End)
```bash
# One command to do everything
python scripts/train_pipeline.py --data data/raw/creditcard.csv
```

This will:
1. Preprocess data
2. Train all models
3. Evaluate ensemble
4. Generate reports

---

## Viewing Results

### MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

View:
- All training runs
- Model metrics
- Parameters
- Artifacts

### Grafana Dashboards
```bash
# After starting docker-compose
# Open http://localhost:3000
# Login: admin/admin
```

View:
- Real-time API metrics
- Request rates
- Latency percentiles
- Fraud detection rates

---

## Troubleshooting

### Dataset Not Found
```bash
# Make sure you downloaded the dataset
ls data/raw/creditcard.csv

# If missing, run:
python scripts/download_data.py
```

### Models Not Found
```bash
# Train the models first
python models/train.py

# Check if models exist
ls models/saved_models/
```

### API Connection Error
```bash
# Check if API is running
curl http://localhost:8000/health

# Check Docker containers
docker ps

# View API logs
docker logs fraud-api
```

### Redis Connection Error
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

---

## Project Structure Reference
```
fraud-detection-system/
├── data/preprocessing.py         # Run first
├── models/train.py               # Run second
├── models/evaluate.py            # Run third
├── api/main.py                   # Start API
├── deployment/docker-compose.yml # Full stack
└── scripts/train_pipeline.py    # All-in-one
```

---

## Common Commands Cheat Sheet
```bash
# Data
python scripts/download_data.py
python data/preprocessing.py

# Training
python models/train.py
python models/evaluate.py

# API
uvicorn api.main:app --reload
docker-compose up -d

# Testing
pytest
python scripts/simulate_traffic.py

# Monitoring
mlflow ui
# Open Grafana at localhost:3000
```

---

## Next Steps

After setup:
1. ✅ Explore notebooks: `jupyter notebook notebooks/`
2. ✅ Read API docs: http://localhost:8000/docs
3. ✅ Check monitoring: http://localhost:3000
4. ✅ Review INTERVIEW_PREP.md for technical questions
5. ✅ Customize for your use case

---

## Need Help?

- **API Docs**: http://localhost:8000/docs
- **Architecture**: docs/ARCHITECTURE.md
- **API Usage**: docs/API_DOCS.md
- **Interview Prep**: docs/INTERVIEW_PREP.md

---

**Estimated Time**:
- Dataset download: 2 min
- Preprocessing: 1 min
- Training: 10 min
- API setup (Docker): 2 min
- **Total**: ~15 minutes to fully operational system!
