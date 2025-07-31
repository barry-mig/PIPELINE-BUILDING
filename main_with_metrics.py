from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Create FastAPI app instance
app = FastAPI(title="FastAPI with Simple Metrics", version="1.0.0")

# Simple Prometheus metrics
request_count = Counter('fastapi_requests_total', 'Total FastAPI requests', ['method', 'endpoint'])
request_duration = Histogram('fastapi_request_duration_seconds', 'FastAPI request duration')

# Simple middleware to track requests
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record metrics
    request_count.labels(method=request.method, endpoint=request.url.path).inc()
    request_duration.observe(duration)
    
    return response

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Basic GET endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello World from FastAPI with Metrics!"}

# GET endpoint with path parameter
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}

# GET endpoint with query parameter
@app.get("/items")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit, "message": "This is a simple items endpoint"}

# POST endpoint
@app.post("/items")
async def create_item(item: dict):
    return {"message": "Item created successfully", "item": item}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "metrics": "enabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
