from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import random

# Create FastAPI app instance
app = FastAPI(title="FastAPI with Monitoring", version="1.0.0")

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'fastapi_requests_total', 
    'Total number of requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'fastapi_request_duration_seconds', 
    'Request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of ML predictions made',
    ['model_version']
)

ERROR_COUNT = Counter(
    'fastapi_errors_total',
    'Total number of errors',
    ['error_type']
)

# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Track request count and duration
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)
    
    return response

# Metrics endpoint for Prometheus to scrape
@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Basic GET endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello from Monitored FastAPI!", "status": "healthy"}

# Health check endpoint with custom metrics
@app.get("/health")
async def health():
    # Simulate some health check logic
    health_status = "ok" if random.random() > 0.1 else "degraded"
    
    if health_status == "degraded":
        ERROR_COUNT.labels(error_type="health_degraded").inc()
    
    return {"status": health_status, "timestamp": time.time()}

# ML Prediction endpoint with metrics
@app.post("/predict")
async def predict(data: dict):
    try:
        # Simulate ML prediction
        prediction_time = random.uniform(0.01, 0.1)  # 10-100ms
        time.sleep(prediction_time)
        
        # Track prediction metrics
        PREDICTION_COUNT.labels(model_version="v1.0").inc()
        
        # Simulate prediction result
        result = {
            "prediction": random.uniform(0, 1),
            "confidence": random.uniform(0.8, 0.99),
            "model_version": "v1.0",
            "processing_time_ms": round(prediction_time * 1000, 2)
        }
        
        return result
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        return {"error": str(e)}, 500

# Load testing endpoint
@app.get("/load")
async def load_test():
    """Endpoint to generate CPU load for testing auto-scaling"""
    start = time.time()
    # CPU intensive task
    while time.time() - start < 0.1:  # 100ms of work
        _ = sum(i * i for i in range(1000))
    
    return {"message": "Load test completed", "duration_ms": 100}

# Simulate different response times
@app.get("/slow")
async def slow_endpoint():
    # Simulate slow operation
    delay = random.uniform(1, 3)
    time.sleep(delay)
    return {"message": f"Slow response after {delay:.2f}s"}

@app.get("/fast")
async def fast_endpoint():
    # Simulate fast operation
    return {"message": "Fast response", "response_time": "< 1ms"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
