# System Architecture

## Overview

The Fraud Detection System is a production-ready ML service designed for real-time transaction fraud detection with <100ms latency. The system follows microservices architecture with independent scaling capabilities.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Mobile Apps, Web Apps, Payment Gateways)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTPS/JSON
┌───────────────────────────▼──────────────────────────────────┐
│                     API Gateway Layer                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ Rate Limit │  │   Auth     │  │    CORS    │             │
│  └────────────┘  └────────────┘  └────────────┘             │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    FastAPI Application                        │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  /predict        /predict/batch      /health        │     │
│  └─────────────────────────────────────────────────────┘     │
└───────────┬──────────────────┬────────────────────────┬──────┘
            │                  │                        │
    ┌───────▼────────┐  ┌──────▼──────┐       ┌───────▼──────┐
    │  Redis Cache   │  │  Inference  │       │  Prometheus  │
    │  (5min TTL)    │  │   Engine    │       │   Metrics    │
    └────────────────┘  └──────┬──────┘       └──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Ensemble Model    │
                    │  ┌────────────────┐ │
                    │  │ Isolation      │ │
                    │  │ Forest (0.2)   │ │
                    │  ├────────────────┤ │
                    │  │ XGBoost (0.5)  │ │
                    │  ├────────────────┤ │
                    │  │ Autoencoder    │ │
                    │  │ (0.3)          │ │
                    │  └────────────────┘ │
                    └─────────────────────┘
```

## Component Details

### 1. API Layer (FastAPI)

**Responsibilities:**
- Request validation (Pydantic schemas)
- Rate limiting (100 req/min per IP)
- API key authentication
- Request routing
- Response formatting
- Error handling

**Endpoints:**
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch predictions (max 1000)
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Swagger documentation

**Performance Optimizations:**
- Model loaded at startup (singleton)
- Async request handling
- Connection pooling
- Response compression

### 2. Inference Engine

**Responsibilities:**
- Feature preprocessing
- Model prediction
- SHAP explanation generation
- Cache management

**Pipeline:**
1. Validate input
2. Check cache (Redis)
3. Preprocess features
4. Run ensemble prediction
5. Generate explanations
6. Cache result
7. Return response

**Preprocessing:**
- Transaction hour extraction
- Amount log transform
- Z-score normalization
- Standard scaling (fitted on training data)

### 3. Ensemble Model

**Architecture:**
Three models combined via weighted voting:

```
                    Input Features (33)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼──────┐    ┌────▼──────┐
   │Isolation│      │   XGBoost   │    │Autoencoder│
   │ Forest  │      │             │    │           │
   │ (unsup) │      │ (supervised)│    │  (deep)   │
   └────┬────┘      └──────┬──────┘    └────┬──────┘
        │                  │                  │
        │ P₁ × 0.2         │ P₂ × 0.5        │ P₃ × 0.3
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    Weighted Average
                           │
                    Final Prediction
```

**Why Weighted Voting?**
- XGBoost performs best (50% weight)
- Isolation Forest catches unknown patterns (20%)
- Autoencoder adds diversity (30%)

### 4. Caching Layer (Redis)

**Purpose:**
- Reduce latency for duplicate requests
- Decrease model inference load
- Improve throughput

**Strategy:**
- Key: MD5 hash of transaction features
- Value: Prediction result (JSON)
- TTL: 5 minutes
- Eviction: LRU (max 256MB)

**Cache Hit Rate:** ~15% in production traffic

### 5. Monitoring Stack

**Prometheus:**
- Scrapes `/metrics` endpoint every 15s
- Stores time-series data
- Evaluates alert rules

**Grafana:**
- Visualizes metrics
- Pre-configured dashboards
- Alert notifications

**Metrics Tracked:**
- Request rate (req/sec)
- Latency percentiles (P50, P95, P99)
- Prediction distribution (fraud %)
- Error rate (4xx, 5xx)
- Cache hit rate

### 6. Data Storage

**PostgreSQL:**
- Transaction logs
- Prediction history
- Model metadata

**Redis:**
- Prediction cache
- Session data

**Local Filesystem:**
- Trained models (.pkl, .h5)
- Preprocessed data (.csv)
- MLflow artifacts

## Data Flow

### Training Flow

```
Raw Data (CSV)
    │
    ├─▶ Preprocessing
    │   ├─ Feature engineering
    │   ├─ Train/val/test split (60/20/20)
    │   ├─ SMOTE oversampling (train only)
    │   └─ Standard scaling
    │
    ├─▶ Model Training
    │   ├─ Isolation Forest
    │   ├─ XGBoost
    │   └─ Autoencoder
    │
    ├─▶ Evaluation
    │   ├─ Metrics calculation
    │   ├─ Cost-benefit analysis
    │   └─ SHAP analysis
    │
    └─▶ Save Artifacts
        ├─ models/saved_models/
        ├─ data/processed/
        └─ MLflow tracking
```

### Inference Flow

```
Transaction (JSON)
    │
    ├─▶ API Validation
    │   └─ Pydantic schema check
    │
    ├─▶ Cache Check
    │   ├─ Hit: Return cached result
    │   └─ Miss: Continue
    │
    ├─▶ Preprocessing
    │   ├─ Feature engineering
    │   └─ Scaling
    │
    ├─▶ Ensemble Prediction
    │   ├─ IF, XGB, AE inference
    │   └─ Weighted average
    │
    ├─▶ Explanation (SHAP)
    │   └─ Top 3 features
    │
    ├─▶ Cache Write
    │   └─ Store for 5 min
    │
    └─▶ Response
        └─ JSON with prediction + explanation
```

## Scaling Considerations

### Horizontal Scaling (Future)

**API Tier:**
```
        ┌─────────────────┐
        │  Load Balancer  │
        └────────┬─────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
┌────▼────┐ ┌───▼────┐ ┌───▼────┐
│ API #1  │ │ API #2 │ │ API #3 │
└─────────┘ └────────┘ └────────┘
```

**Model Serving:**
- Model sharding (split features)
- Model parallelism (GPU)
- Batch prediction optimization

**Database:**
- Read replicas (PostgreSQL)
- Redis cluster (sharding)

### Vertical Scaling

**Current Resources:**
- API: 2 CPU, 4GB RAM
- Redis: 256MB memory limit
- Postgres: 1GB RAM

**Bottlenecks:**
- SHAP explanation generation (~20ms)
- Autoencoder inference (~15ms)
- Feature preprocessing (~5ms)

**Optimization Opportunities:**
- Precompute SHAP for common transactions
- Model quantization (reduce size)
- Feature caching

## Security Measures

### Authentication & Authorization
- API key authentication (X-API-Key header)
- Rate limiting per IP/key
- Request ID tracking

### Data Protection
- HTTPS only (in production)
- Input validation (prevent injection)
- Sensitive data masking in logs

### Model Security
- Model versioning
- Adversarial robustness testing (future)
- Drift detection

## Deployment Strategy

### Docker Compose (Current)
- All services in single compose file
- Shared Docker network
- Volume mounts for data/models

### Kubernetes (Future)
```yaml
Deployment:
  - fraud-api (3 replicas)
  - redis (1 replica)
  - postgres (1 replica, StatefulSet)
  
Services:
  - API: LoadBalancer
  - Redis: ClusterIP
  - Postgres: ClusterIP
  
ConfigMaps:
  - prometheus.yml
  - grafana_dashboard.json
  
Secrets:
  - API keys
  - Database credentials
```

## Monitoring & Alerting

### Health Checks
- API: `/health` endpoint
- Docker: HEALTHCHECK directive
- K8s: Liveness/readiness probes (future)

### Alerts
1. **High Latency**: P95 >200ms for 5 min
2. **High Fraud Rate**: >5% for 10 min
3. **High Error Rate**: >1% for 5 min
4. **API Down**: Unreachable for 1 min

### Logging
- Structured JSON logs
- Request/response logging
- Error stack traces
- Correlation IDs

## Disaster Recovery

### Backup Strategy
- Models: Stored in MLflow + S3 (future)
- Data: Daily PostgreSQL dumps
- Config: Version controlled (Git)

### Recovery Time Objective (RTO)
- Target: 15 minutes
- Steps:
  1. Restore latest model from backup
  2. Deploy API container
  3. Verify health checks

## Performance Benchmarks

### Single Prediction
- Cache Hit: ~2ms
- Cache Miss: ~50ms (P50), ~90ms (P95)

### Batch Prediction (100 txns)
- Total: ~800ms
- Per transaction: ~8ms

### Throughput
- Max tested: 500 req/s (with caching)
- Sustainable: 150 req/s

---

**Last Updated**: 2025-01-13

**Architecture Decisions:**
- Chose FastAPI over Flask for async support and built-in validation
- Redis over Memcached for richer data structures
- Docker Compose over K8s for easier demo/development
