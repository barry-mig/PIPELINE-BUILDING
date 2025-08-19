# ===================================================================
# PRODUCTION DEPLOYMENT GUIDE
# ===================================================================
# Complete guide for deploying the Customer Churn Prediction API
# to production environments including cloud platforms, monitoring,
# and operational best practices.
# ===================================================================

# 🚀 PRODUCTION DEPLOYMENT GUIDE FOR ML PIPELINE API

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Platform Deployment](#cloud-platform-deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Security Configuration](#security-configuration)
9. [Performance Optimization](#performance-optimization)
10. [Operational Procedures](#operational-procedures)
11. [Troubleshooting](#troubleshooting)

---

## 🔍 Pre-Deployment Checklist

### ✅ Code Readiness
- [ ] All tests passing (unit, integration, performance)
- [ ] Code review completed and approved
- [ ] Security audit completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] API contracts finalized

### ✅ Infrastructure Readiness
- [ ] Production environment provisioned
- [ ] Database setup completed
- [ ] Monitoring stack deployed
- [ ] Load balancers configured
- [ ] SSL certificates installed
- [ ] Backup systems configured

### ✅ Data Readiness
- [ ] ML models trained and validated
- [ ] Data pipelines tested
- [ ] Data quality checks implemented
- [ ] Feature stores configured
- [ ] Model registry setup

### ✅ Security Readiness
- [ ] Authentication mechanisms configured
- [ ] API keys generated and distributed
- [ ] Network security rules applied
- [ ] Secrets management configured
- [ ] Compliance requirements met

---

## 🌍 Environment Setup

### Development Environment
```bash
# Create virtual environment
python -m venv ml_pipeline_env
source ml_pipeline_env/bin/activate  # On Windows: ml_pipeline_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env.local
# Edit .env.local with your local settings

# Run locally
python ml_pipeline_api.py
```

### Staging Environment
```bash
# Clone repository
git clone https://github.com/your-org/ml-pipeline-api.git
cd ml-pipeline-api

# Setup staging environment
cp .env.example .env.staging
# Configure staging-specific settings

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d
```

### Production Environment
```bash
# Production deployment
cp .env.example .env.production
# Configure production settings with secure values

# Deploy using your preferred method (Docker, Kubernetes, etc.)
```

---

## 🗄️ Database Configuration

### PostgreSQL Setup (Recommended)
```sql
-- Create database and user
CREATE DATABASE customer_churn_db;
CREATE USER ml_api_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE customer_churn_db TO ml_api_user;

-- Create tables for customer data
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    age INTEGER NOT NULL,
    gender VARCHAR(10) NOT NULL,
    income DECIMAL(10, 2) NOT NULL,
    tenure_months INTEGER NOT NULL,
    monthly_charges DECIMAL(8, 2) NOT NULL,
    total_charges DECIMAL(10, 2) NOT NULL,
    phone_service BOOLEAN NOT NULL,
    internet_service VARCHAR(20) NOT NULL,
    streaming_tv BOOLEAN NOT NULL,
    streaming_movies BOOLEAN NOT NULL,
    tech_support BOOLEAN NOT NULL,
    device_protection BOOLEAN NOT NULL,
    contract_type VARCHAR(20) NOT NULL,
    paperless_billing BOOLEAN NOT NULL,
    payment_method VARCHAR(30) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_customers_age ON customers(age);
CREATE INDEX idx_customers_tenure ON customers(tenure_months);
CREATE INDEX idx_customers_charges ON customers(monthly_charges);
CREATE INDEX idx_customers_contract ON customers(contract_type);

-- Create table for prediction history
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    churn_probability DECIMAL(5, 4) NOT NULL,
    risk_category VARCHAR(20) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create table for batch jobs
CREATE TABLE batch_jobs (
    batch_id VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) NOT NULL,
    total_customers INTEGER NOT NULL,
    processed_customers INTEGER DEFAULT 0,
    completed_customers INTEGER DEFAULT 0,
    failed_customers INTEGER DEFAULT 0,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    priority VARCHAR(20) DEFAULT 'normal'
);
```

### Redis Setup (for caching)
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis                  # macOS

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set password: requirepass your_redis_password
# Set memory policy: maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash ml_user
RUN chown -R ml_user:ml_user /app
USER ml_user

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["python", "ml_pipeline_api.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://ml_user:password@postgres:5432/customer_churn_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    networks:
      - ml-network

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=customer_churn_db
      - POSTGRES_USER=ml_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass password
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - ml-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped
    networks:
      - ml-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  ml-network:
    driver: bridge
```

### Build and Deploy
```bash
# Build the Docker image
docker build -t ml-pipeline-api:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f ml-api

# Scale the service
docker-compose up -d --scale ml-api=3
```

---

## ☸️ Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-pipeline
  labels:
    name: ml-pipeline
    environment: production
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: ml-pipeline
data:
  APP_NAME: "Customer Churn Prediction API"
  ENVIRONMENT: "production"
  LOG_LEVEL: "info"
  WORKERS: "4"
  HOST: "0.0.0.0"
  PORT: "8001"
```

### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secrets
  namespace: ml-pipeline
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_PASSWORD: <base64-encoded-redis-password>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-pipeline
  labels:
    app: ml-api
    version: v2.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        version: v2.0.0
    spec:
      containers:
      - name: ml-api
        image: your-registry/ml-pipeline-api:v2.0.0
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: ml-api-config
        - secretRef:
            name: ml-api-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
```

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
  namespace: ml-pipeline
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: ClusterIP
```

### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: ml-pipeline
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.yourcompany.com
    secretName: ml-api-tls
  rules:
  - host: api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-api-service
            port:
              number: 80
```

### HorizontalPodAutoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: ml-pipeline
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods -n ml-pipeline
kubectl get services -n ml-pipeline
kubectl get ingress -n ml-pipeline

# View logs
kubectl logs -f deployment/ml-api -n ml-pipeline
```

---

## ☁️ Cloud Platform Deployment

### AWS Deployment

#### Using AWS ECS
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name ml-pipeline-cluster

# Build and push Docker image to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker build -t ml-pipeline-api .
docker tag ml-pipeline-api:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/ml-pipeline-api:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/ml-pipeline-api:latest

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service --cluster ml-pipeline-cluster --service-name ml-api-service --task-definition ml-api:1 --desired-count 3
```

#### Using AWS Lambda (Serverless)
```python
# lambda_handler.py
import json
from mangum import Mangum
from ml_pipeline_api import app

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Platform

#### Using Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-pipeline-api
gcloud run deploy ml-api --image gcr.io/PROJECT_ID/ml-pipeline-api --platform managed --region us-central1
```

### Microsoft Azure

#### Using Azure Container Instances
```bash
# Create resource group
az group create --name ml-pipeline-rg --location eastus

# Deploy container
az container create --resource-group ml-pipeline-rg --name ml-api --image your-registry/ml-pipeline-api:latest --cpu 2 --memory 4 --ports 8001
```

---

## 📊 Monitoring and Observability

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ml_api_rules.yml"

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "ML Pipeline API Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_api_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      },
      {
        "title": "Predictions per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[1m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules
```yaml
# ml_api_rules.yml
groups:
- name: ml_api_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(ml_api_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(ml_api_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: ModelAccuracyDrop
    expr: ml_model_accuracy < 0.8
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Model accuracy dropped below threshold"
      description: "Current model accuracy is {{ $value }}"
```

---

## 🔐 Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificate with Let's Encrypt
sudo certbot --nginx -d api.yourcompany.com

# Or use a wildcard certificate
sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials ~/.secrets/certbot/cloudflare.ini -d *.yourcompany.com
```

### API Security Headers
```python
# Add to FastAPI middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Enforce HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### Network Security
```bash
# Firewall rules (iptables)
sudo iptables -A INPUT -p tcp --dport 8001 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8001 -j DROP

# Or using cloud provider security groups
# AWS Security Group rules:
# - Allow HTTPS (443) from 0.0.0.0/0
# - Allow HTTP (80) from 0.0.0.0/0 (for redirect)
# - Allow API (8001) from load balancer only
# - Allow SSH (22) from admin IPs only
```

---

## ⚡ Performance Optimization

### Application Optimization
```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Implement caching
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def get_model_prediction(customer_features_hash):
    # Cache expensive model predictions
    pass

# Use async for I/O operations
import asyncio
import aiohttp

async def make_external_api_call():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://external-api.com/data') as response:
            return await response.json()
```

### Database Optimization
```sql
-- Create appropriate indexes
CREATE INDEX CONCURRENTLY idx_customers_composite ON customers(age, tenure_months, monthly_charges);
CREATE INDEX CONCURRENTLY idx_prediction_history_customer_date ON prediction_history(customer_id, predicted_at);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM customers WHERE age BETWEEN 25 AND 45 AND tenure_months > 12;

-- Use read replicas for analytics
-- Configure read-only connection for reporting queries
```

### Infrastructure Optimization
```yaml
# Kubernetes resource optimization
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# Use node affinity for performance
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: instance-type
          operator: In
          values: ["c5.2xlarge", "c5.4xlarge"]
```

---

## 🔧 Operational Procedures

### Deployment Process
```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

echo "Starting production deployment..."

# 1. Backup current version
kubectl create backup production-backup-$(date +%Y%m%d-%H%M%S)

# 2. Update application
kubectl set image deployment/ml-api ml-api=ml-pipeline-api:v2.0.1 -n ml-pipeline

# 3. Wait for rollout
kubectl rollout status deployment/ml-api -n ml-pipeline --timeout=300s

# 4. Run health checks
curl -f https://api.yourcompany.com/health

# 5. Run smoke tests
python tests/smoke_test.py --env production

# 6. Update monitoring
echo "Deployment completed successfully"
```

### Rollback Process
```bash
#!/bin/bash
# rollback.sh - Emergency rollback script

echo "Starting emergency rollback..."

# Get previous revision
PREVIOUS_REVISION=$(kubectl rollout history deployment/ml-api -n ml-pipeline | tail -2 | head -1 | awk '{print $1}')

# Rollback
kubectl rollout undo deployment/ml-api --to-revision=$PREVIOUS_REVISION -n ml-pipeline

# Wait for rollback
kubectl rollout status deployment/ml-api -n ml-pipeline --timeout=300s

# Verify health
curl -f https://api.yourcompany.com/health

echo "Rollback completed"
```

### Backup Procedures
```bash
#!/bin/bash
# backup.sh - Database backup script

# Database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d_%H%M%S).sql s3://your-backup-bucket/ml-api/

# Model backup
tar -czf models_$(date +%Y%m%d_%H%M%S).tar.gz /app/models/
aws s3 cp models_$(date +%Y%m%d_%H%M%S).tar.gz s3://your-backup-bucket/ml-api/models/

# Clean up old backups (keep 30 days)
find . -name "backup_*.sql" -mtime +30 -delete
aws s3 ls s3://your-backup-bucket/ml-api/ | grep backup_ | awk '$1 < "'$(date -d '30 days ago' +%Y-%m-%d)'"' | awk '{print $4}' | xargs -I {} aws s3 rm s3://your-backup-bucket/ml-api/{}
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n ml-pipeline

# Investigate memory leaks
kubectl exec -it pod/ml-api-xxx -n ml-pipeline -- /bin/bash
pip install memory-profiler
python -m memory_profiler ml_pipeline_api.py

# Solution: Optimize memory usage
# - Implement proper connection pooling
# - Clear caches periodically
# - Use pagination for large datasets
```

#### Database Connection Issues
```bash
# Check connection pool
SELECT * FROM pg_stat_activity WHERE datname = 'customer_churn_db';

# Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';

# Solution: 
# - Increase connection pool size
# - Optimize slow queries
# - Implement connection retry logic
```

#### Model Performance Degradation
```python
# Monitor model metrics
def check_model_performance():
    recent_predictions = get_recent_predictions(days=7)
    accuracy = calculate_accuracy(recent_predictions)
    
    if accuracy < 0.8:
        send_alert("Model accuracy dropped to {:.2f}".format(accuracy))
        
# Set up automated retraining
if model_drift_detected():
    trigger_model_retraining()
```

### Debugging Tools
```bash
# Application logs
kubectl logs -f deployment/ml-api -n ml-pipeline

# Performance profiling
pip install py-spy
py-spy top --pid $(pgrep -f ml_pipeline_api)

# Database performance
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM customers WHERE age > 50;

# Network debugging
kubectl exec -it pod/ml-api-xxx -n ml-pipeline -- nslookup postgres-service
kubectl exec -it pod/ml-api-xxx -n ml-pipeline -- telnet postgres-service 5432
```

---

## 📈 Performance Monitoring

### Key Metrics to Monitor

1. **Application Metrics**
   - Request rate (requests/second)
   - Response time (p50, p95, p99)
   - Error rate (%)
   - Prediction accuracy
   - Model inference time

2. **Infrastructure Metrics**
   - CPU utilization (%)
   - Memory usage (%)
   - Disk I/O
   - Network bandwidth
   - Database connections

3. **Business Metrics**
   - Predictions per day
   - High-risk customers identified
   - Retention campaign effectiveness
   - API adoption rate

### Alerting Thresholds
```yaml
# alerts.yml
critical_alerts:
  - error_rate > 5%
  - response_time_p95 > 1000ms
  - model_accuracy < 80%
  - cpu_usage > 90%
  - memory_usage > 95%

warning_alerts:
  - error_rate > 1%
  - response_time_p95 > 500ms
  - model_accuracy < 85%
  - cpu_usage > 70%
  - memory_usage > 80%
```

---

## 🔄 Continuous Integration/Continuous Deployment (CI/CD)

### GitHub Actions Workflow
```yaml
name: ML Pipeline API CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run security scan
      run: |
        pip install safety bandit
        safety check
        bandit -r . -f json -o bandit-report.json

  build-and-deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: |
        docker build -t ml-pipeline-api:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ml-pipeline-api:${{ github.sha }}
    - name: Deploy to production
      run: |
        kubectl set image deployment/ml-api ml-api=ml-pipeline-api:${{ github.sha }} -n ml-pipeline
```

---

## 📞 Support and Maintenance

### Contact Information
- **Technical Support**: ml-team@yourcompany.com
- **Emergency Hotline**: +1-555-ML-HELP
- **Documentation**: https://docs.yourcompany.com/ml-api
- **Status Page**: https://status.yourcompany.com

### Maintenance Windows
- **Regular Maintenance**: Sundays 2:00-4:00 AM UTC
- **Emergency Maintenance**: As needed with 1-hour notice
- **Model Updates**: Deployed during regular maintenance windows

### Support Escalation
1. **Level 1**: Application issues, user support
2. **Level 2**: Infrastructure issues, performance problems
3. **Level 3**: Security incidents, critical failures

---

This comprehensive deployment guide provides everything needed to successfully deploy and operate the ML Pipeline API in production. Follow the checklist, customize the configurations for your environment, and establish proper monitoring and operational procedures for a robust production deployment.
