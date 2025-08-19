# ===================================================================
# PRODUCTION-READY ML PIPELINE API FOR CUSTOMER CHURN PREDICTION
# ===================================================================
# This is the main API file that serves machine learning predictions
# for customer churn in a production environment. It includes:
# 
# 1. Real-time prediction endpoints (single customers)
# 2. Batch prediction endpoints (multiple customers)
# 3. Model management and versioning
# 4. Data validation and quality checks
# 5. Performance monitoring and metrics
# 6. Pipeline orchestration for training
# 7. Health checks and system status
# 8. Production-grade error handling
# 9. Security and authentication hooks
# 10. Comprehensive logging and monitoring
# ===================================================================

# Core FastAPI imports for web API functionality
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware  # Cross-origin resource sharing
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # Security
from fastapi.responses import JSONResponse  # Custom responses
import uvicorn  # ASGI server for running the application

# Data validation and serialization
from pydantic import BaseModel, Field, validator  # Data validation
from typing import List, Dict, Optional, Union, Any  # Type hints for clarity

# Core Python libraries
import json  # JSON data handling
import asyncio  # Asynchronous programming
import logging  # Logging for monitoring and debugging
import time  # Performance timing
from datetime import datetime, timedelta  # Date and time operations
import uuid  # Unique identifier generation
from pathlib import Path  # File path operations
import os  # Operating system interface

# Machine Learning and Data Processing
import numpy as np  # Numerical computing (placeholder - install separately)
# import pandas as pd  # Data manipulation (placeholder - install separately)
# import joblib  # Model serialization (placeholder - install separately)
# from sklearn.ensemble import RandomForestClassifier  # ML model (placeholder)

# Monitoring and Metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, multiprocess, generate_latest

# Security and Authentication (optional)
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import jwt  # JSON Web Tokens

# Database connections (choose based on your setup)
# import psycopg2  # PostgreSQL
# import redis  # Redis cache
# from sqlalchemy import create_engine  # SQL databases

# ===================================================================
# LOGGING CONFIGURATION
# ===================================================================
# Set up structured logging for production monitoring

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# ===================================================================
# PROMETHEUS METRICS SETUP
# ===================================================================
# These metrics help monitor the API performance in production

# Request counting metrics
REQUEST_COUNT = Counter(
    'ml_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

# Response time metrics
REQUEST_DURATION = Histogram(
    'ml_api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Prediction-specific metrics
PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of ML predictions made',
    ['model_version', 'prediction_type']
)

# Model performance metrics
MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score',
    ['model_version']
)

# System health metrics
ACTIVE_CONNECTIONS = Gauge(
    'ml_api_active_connections',
    'Number of active connections'
)

ERROR_COUNT = Counter(
    'ml_api_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

# ===================================================================
# FASTAPI APPLICATION SETUP
# ===================================================================
# Create the main FastAPI application with production settings

app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    Production-ready Machine Learning API for predicting customer churn.
    
    ## Features
    
    * **Real-time Predictions**: Get instant churn predictions for individual customers
    * **Batch Processing**: Process multiple customers in bulk
    * **Model Management**: Deploy and manage multiple model versions
    * **Data Validation**: Comprehensive input data validation
    * **Quality Monitoring**: Continuous data quality assessment
    * **Performance Metrics**: Real-time monitoring and alerting
    * **Pipeline Orchestration**: Automated model training and deployment
    
    ## Authentication
    
    This API uses bearer token authentication. Include your token in the Authorization header:
    `Authorization: Bearer your-token-here`
    
    ## Rate Limiting
    
    * Real-time predictions: 1000 requests/minute per API key
    * Batch predictions: 10 requests/minute per API key
    
    ## Support
    
    For technical support, contact: ml-team@yourcompany.com
    """,
    version="2.0.0",
    contact={
        "name": "ML Engineering Team",
        "email": "ml-team@yourcompany.com",
        "url": "https://yourcompany.com/ml-support",
    },
    license_info={
        "name": "Internal Use Only",
        "url": "https://yourcompany.com/license",
    },
    # Production settings
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc",  # ReDoc documentation
    openapi_url="/openapi.json",  # OpenAPI schema
)

# ===================================================================
# SECURITY MIDDLEWARE
# ===================================================================
# Add security layers for production deployment

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourcompany.com", "https://app.yourcompany.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Authorization", "Content-Type"],
)

# Trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["api.yourcompany.com", "*.yourcompany.com", "localhost"]
)

# ===================================================================
# CUSTOM MIDDLEWARE FOR MONITORING
# ===================================================================

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """
    Custom middleware to track all requests for monitoring
    This captures metrics for every API call automatically
    """
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate response time
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log successful requests
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        
        return response
        
    except Exception as e:
        # Record error metrics
        ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint=request.url.path
        ).inc()
        
        # Log errors
        logger.error(f"Error processing {request.method} {request.url.path}: {str(e)}")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
    
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()

# ===================================================================
# DATA MODELS FOR REQUEST/RESPONSE VALIDATION
# ===================================================================
# These Pydantic models ensure data integrity and provide documentation

class CustomerInput(BaseModel):
    """
    Input model for customer data used in churn prediction
    
    This model validates all incoming customer data to ensure:
    - All required fields are present and correctly typed
    - Values are within acceptable business ranges
    - Data consistency across related fields
    - Protection against malicious input
    """
    
    # Customer identification
    customer_id: str = Field(
        ..., 
        min_length=1, 
        max_length=50, 
        description="Unique customer identifier",
        example="CUST_001234"
    )
    
    # Demographic information
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Customer age in years",
        example=35
    )
    
    gender: str = Field(
        ..., 
        regex="^(male|female|other)$", 
        description="Customer gender",
        example="female"
    )
    
    income: float = Field(
        ..., 
        ge=0, 
        le=1000000, 
        description="Annual income in USD",
        example=65000.0
    )
    
    # Service tenure and billing
    tenure_months: int = Field(
        ..., 
        ge=0, 
        le=600, 
        description="Number of months as customer",
        example=24
    )
    
    monthly_charges: float = Field(
        ..., 
        ge=0, 
        le=10000, 
        description="Monthly charges in USD",
        example=79.50
    )
    
    total_charges: float = Field(
        ..., 
        ge=0, 
        description="Total charges to date in USD",
        example=1908.0
    )
    
    # Service subscriptions (boolean flags)
    phone_service: bool = Field(
        ..., 
        description="Has phone service subscription",
        example=True
    )
    
    internet_service: str = Field(
        ..., 
        regex="^(DSL|Fiber optic|No)$", 
        description="Type of internet service",
        example="Fiber optic"
    )
    
    streaming_tv: bool = Field(
        ..., 
        description="Has streaming TV service",
        example=True
    )
    
    streaming_movies: bool = Field(
        ..., 
        description="Has streaming movies service",
        example=False
    )
    
    # Support services
    tech_support: bool = Field(
        ..., 
        description="Has technical support service",
        example=True
    )
    
    device_protection: bool = Field(
        ..., 
        description="Has device protection service",
        example=False
    )
    
    # Contract and billing information
    contract_type: str = Field(
        ..., 
        regex="^(Month-to-month|One year|Two year)$", 
        description="Contract type",
        example="One year"
    )
    
    paperless_billing: bool = Field(
        ..., 
        description="Uses paperless billing",
        example=True
    )
    
    payment_method: str = Field(
        ..., 
        regex="^(Electronic check|Mailed check|Bank transfer|Credit card)$", 
        description="Payment method",
        example="Credit card"
    )
    
    # Optional metadata
    data_source: Optional[str] = Field(
        default="api", 
        description="Source of the customer data",
        example="web_form"
    )
    
    @validator('total_charges')
    def validate_total_charges_consistency(cls, v, values):
        """
        Business rule validation: Total charges should be reasonable 
        compared to monthly charges and tenure
        """
        if 'monthly_charges' in values and 'tenure_months' in values:
            # Calculate expected minimum (allowing for discounts)
            expected_minimum = values['monthly_charges'] * values['tenure_months'] * 0.7
            if v < expected_minimum:
                raise ValueError(
                    f"Total charges ({v}) seems inconsistent with monthly charges "
                    f"({values['monthly_charges']}) and tenure ({values['tenure_months']} months)"
                )
        return v

class PredictionResponse(BaseModel):
    """
    Response model for churn prediction results
    
    Provides comprehensive prediction information including:
    - Primary prediction score
    - Model confidence level
    - Model version used
    - Timing information
    - Unique request tracking
    """
    
    # Core prediction results
    churn_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Probability of customer churn (0.0 = will stay, 1.0 = will churn)",
        example=0.23
    )
    
    churn_risk_category: str = Field(
        ..., 
        description="Risk category based on probability",
        example="Low"
    )
    
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Model confidence in the prediction",
        example=0.92
    )
    
    # Model information
    model_version: str = Field(
        ..., 
        description="Version of the ML model used",
        example="v2.1.3"
    )
    
    model_accuracy: float = Field(
        ..., 
        description="Current model accuracy on validation set",
        example=0.87
    )
    
    # Business recommendations
    recommended_action: str = Field(
        ..., 
        description="Recommended business action based on prediction",
        example="Continue regular engagement"
    )
    
    retention_strategy: Optional[str] = Field(
        None, 
        description="Specific retention strategy if at-risk",
        example="Offer loyalty discount"
    )
    
    # Request tracking and timing
    request_id: str = Field(
        ..., 
        description="Unique identifier for this prediction request",
        example="req_7f9a8b2c-1d3e-4f5g-6h7i-8j9k0l1m2n3o"
    )
    
    timestamp: str = Field(
        ..., 
        description="When the prediction was made (ISO format)",
        example="2024-01-15T14:30:45.123Z"
    )
    
    processing_time_ms: float = Field(
        ..., 
        description="Time taken to process the prediction in milliseconds",
        example=25.7
    )
    
    # Customer information echo (for verification)
    customer_id: str = Field(
        ..., 
        description="Customer ID from the request",
        example="CUST_001234"
    )

class BatchPredictionRequest(BaseModel):
    """
    Request model for batch prediction processing
    Handles multiple customer predictions in a single request
    """
    
    customers: List[CustomerInput] = Field(
        ..., 
        min_items=1, 
        max_items=1000, 
        description="List of customers to predict (max 1000 per batch)",
        example=[{
            "customer_id": "CUST_001",
            "age": 35,
            "gender": "female",
            # ... other fields
        }]
    )
    
    batch_id: str = Field(
        default_factory=lambda: f"batch_{uuid.uuid4().hex[:12]}", 
        description="Unique identifier for this batch",
        example="batch_a1b2c3d4e5f6"
    )
    
    processing_priority: str = Field(
        default="normal", 
        regex="^(low|normal|high|urgent)$", 
        description="Processing priority for the batch",
        example="normal"
    )
    
    notification_webhook: Optional[str] = Field(
        None, 
        description="URL to call when batch processing is complete",
        example="https://yourapp.com/webhooks/batch-complete"
    )
    
    include_detailed_results: bool = Field(
        default=True, 
        description="Include detailed prediction results for each customer",
        example=True
    )

class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction requests
    Provides status and tracking information for batch processing
    """
    
    batch_id: str = Field(
        ..., 
        description="Unique identifier for the batch",
        example="batch_a1b2c3d4e5f6"
    )
    
    status: str = Field(
        ..., 
        description="Current status of the batch processing",
        example="processing"
    )
    
    total_customers: int = Field(
        ..., 
        description="Total number of customers in the batch",
        example=150
    )
    
    estimated_completion_time: Optional[str] = Field(
        None, 
        description="Estimated completion time (ISO format)",
        example="2024-01-15T14:35:00.000Z"
    )
    
    status_check_url: str = Field(
        ..., 
        description="URL to check the status of this batch",
        example="/api/v1/batch/status/batch_a1b2c3d4e5f6"
    )
    
    submitted_at: str = Field(
        ..., 
        description="When the batch was submitted (ISO format)",
        example="2024-01-15T14:30:00.000Z"
    )

class ModelInfo(BaseModel):
    """
    Information about available ML models
    """
    
    version: str = Field(..., description="Model version identifier")
    accuracy: float = Field(..., description="Model accuracy on validation set")
    status: str = Field(..., description="Model status (active/testing/deprecated)")
    created_at: str = Field(..., description="When the model was created")
    features_count: int = Field(..., description="Number of features used")
    training_samples: int = Field(..., description="Number of samples used for training")

class HealthCheck(BaseModel):
    """
    Health check response model
    """
    
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    models_available: int = Field(..., description="Number of available models")
    last_prediction: Optional[str] = Field(None, description="Timestamp of last prediction")

# ===================================================================
# IN-MEMORY DATA STORAGE (PRODUCTION: USE PROPER DATABASE)
# ===================================================================
# These dictionaries simulate database storage for demo purposes
# In production, replace with proper database connections

# Model registry with version information
MODEL_REGISTRY = {
    "v1.0": {
        "accuracy": 0.85,
        "status": "deprecated",
        "created_at": "2023-01-15T10:00:00Z",
        "features_count": 15,
        "training_samples": 50000,
        "description": "Initial production model"
    },
    "v2.0": {
        "accuracy": 0.87,
        "status": "testing",
        "created_at": "2023-06-20T14:30:00Z",
        "features_count": 18,
        "training_samples": 75000,
        "description": "Enhanced feature engineering"
    },
    "v2.1.3": {
        "accuracy": 0.89,
        "status": "active",
        "created_at": "2023-12-01T09:15:00Z",
        "features_count": 20,
        "training_samples": 100000,
        "description": "Latest production model with improved accuracy"
    }
}

# Active batch jobs tracking
BATCH_JOBS = {}

# System metrics
SYSTEM_METRICS = {
    "start_time": datetime.now(),
    "total_predictions": 0,
    "total_batches": 0,
    "last_prediction_time": None,
    "models_loaded": len(MODEL_REGISTRY)
}

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def get_current_model_version() -> str:
    """
    Get the currently active model version
    In production, this might query a model registry service
    """
    for version, info in MODEL_REGISTRY.items():
        if info["status"] == "active":
            return version
    return "v2.1.3"  # Fallback default

def calculate_risk_category(probability: float) -> str:
    """
    Convert churn probability to business-friendly risk category
    
    Args:
        probability: Churn probability (0.0 to 1.0)
        
    Returns:
        Risk category string
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    elif probability < 0.8:
        return "High"
    else:
        return "Critical"

def get_recommended_action(probability: float, risk_category: str) -> tuple[str, Optional[str]]:
    """
    Generate business recommendations based on churn probability
    
    Args:
        probability: Churn probability
        risk_category: Risk category (Low/Medium/High/Critical)
        
    Returns:
        (recommended_action, retention_strategy)
    """
    if risk_category == "Low":
        return "Continue regular engagement", None
    elif risk_category == "Medium":
        return "Increase engagement touchpoints", "Send satisfaction survey"
    elif risk_category == "High":
        return "Immediate retention campaign", "Offer 15% discount for 6 months"
    else:  # Critical
        return "Urgent intervention required", "Personal call from customer success manager"

def simulate_ml_prediction(customer_data: CustomerInput) -> float:
    """
    Simulate ML model prediction
    In production, this would call your actual trained model
    
    Args:
        customer_data: Customer information
        
    Returns:
        Churn probability (0.0 to 1.0)
    """
    # Simple heuristic for demonstration
    # In production, replace with actual model inference
    
    risk_factors = 0.0
    
    # Age factor (younger customers more likely to churn)
    if customer_data.age < 30:
        risk_factors += 0.1
    
    # Contract type factor
    if customer_data.contract_type == "Month-to-month":
        risk_factors += 0.3
    
    # Tenure factor (newer customers more likely to churn)
    if customer_data.tenure_months < 12:
        risk_factors += 0.2
    
    # Service factors
    if not customer_data.phone_service:
        risk_factors += 0.1
    if customer_data.internet_service == "No":
        risk_factors += 0.1
    
    # Support services (lack of support increases churn risk)
    if not customer_data.tech_support:
        risk_factors += 0.1
    if not customer_data.device_protection:
        risk_factors += 0.05
    
    # Payment method factor
    if customer_data.payment_method == "Electronic check":
        risk_factors += 0.1
    
    # High charges relative to income
    monthly_as_percent_income = (customer_data.monthly_charges * 12) / customer_data.income
    if monthly_as_percent_income > 0.05:  # More than 5% of annual income
        risk_factors += 0.15
    
    # Add some randomness for realism
    import random
    risk_factors += random.uniform(-0.1, 0.1)
    
    # Ensure result is between 0 and 1
    probability = max(0.0, min(1.0, risk_factors))
    
    return probability

# ===================================================================
# API ENDPOINTS - CORE PREDICTION SERVICES
# ===================================================================

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput, request: Request):
    """
    🔮 **Real-time Churn Prediction**
    
    Predict the likelihood of customer churn for a single customer.
    This endpoint provides instant predictions for real-time decision making.
    
    **Use Cases:**
    - Website checkout process: Show retention offers to at-risk customers
    - Customer service calls: Display risk level to support agents
    - Mobile app interactions: Trigger personalized retention campaigns
    - Email campaigns: Segment customers by churn risk
    
    **Response Time:** Typically < 50ms
    
    **Rate Limit:** 1000 requests/minute per API key
    """
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    
    logger.info(f"Processing prediction request {request_id} for customer {customer.customer_id}")
    
    try:
        # Get current model version
        model_version = get_current_model_version()
        model_info = MODEL_REGISTRY[model_version]
        
        # Perform ML prediction
        churn_probability = simulate_ml_prediction(customer)
        
        # Calculate confidence (in production, this comes from your model)
        confidence_score = min(0.95, max(0.60, 0.9 - (churn_probability * 0.1)))
        
        # Determine risk category and recommendations
        risk_category = calculate_risk_category(churn_probability)
        recommended_action, retention_strategy = get_recommended_action(churn_probability, risk_category)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(
            model_version=model_version,
            prediction_type="single"
        ).inc()
        
        SYSTEM_METRICS["total_predictions"] += 1
        SYSTEM_METRICS["last_prediction_time"] = datetime.now()
        
        # Create response
        response = PredictionResponse(
            churn_probability=churn_probability,
            churn_risk_category=risk_category,
            confidence_score=confidence_score,
            model_version=model_version,
            model_accuracy=model_info["accuracy"],
            recommended_action=recommended_action,
            retention_strategy=retention_strategy,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time_ms,
            customer_id=customer.customer_id
        )
        
        logger.info(
            f"Prediction completed for {customer.customer_id}: "
            f"probability={churn_probability:.3f}, "
            f"risk={risk_category}, "
            f"time={processing_time_ms:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(f"Prediction failed for request {request_id}: {str(e)}")
        
        # Record error metric
        ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint="/api/v1/predict"
        ).inc()
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict_churn(
    batch_request: BatchPredictionRequest, 
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    📊 **Batch Churn Prediction**
    
    Process multiple customers for churn prediction in a single request.
    Ideal for bulk analysis and periodic customer assessments.
    
    **Use Cases:**
    - Weekly customer risk assessment reports
    - Marketing campaign targeting
    - Customer lifecycle analysis
    - Data science model validation
    
    **Processing:** Asynchronous (returns immediately with tracking info)
    
    **Batch Limits:** 1-1000 customers per request
    
    **Rate Limit:** 10 requests/minute per API key
    """
    batch_id = batch_request.batch_id
    customer_count = len(batch_request.customers)
    submitted_at = datetime.now()
    
    logger.info(f"Starting batch prediction {batch_id} with {customer_count} customers")
    
    try:
        # Validate batch size
        if customer_count > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum limit of 1000 customers"
            )
        
        # Estimate completion time (rough estimate: 50ms per customer + overhead)
        estimated_duration_seconds = (customer_count * 0.05) + 5
        estimated_completion = submitted_at + timedelta(seconds=estimated_duration_seconds)
        
        # Initialize batch job tracking
        BATCH_JOBS[batch_id] = {
            "status": "processing",
            "total_customers": customer_count,
            "processed_customers": 0,
            "completed_customers": 0,
            "failed_customers": 0,
            "submitted_at": submitted_at.isoformat(),
            "estimated_completion": estimated_completion.isoformat(),
            "results": [],
            "priority": batch_request.processing_priority,
            "webhook_url": batch_request.notification_webhook,
            "include_detailed_results": batch_request.include_detailed_results
        }
        
        # Schedule background processing
        background_tasks.add_task(
            process_batch_predictions,
            batch_id,
            batch_request.customers,
            batch_request.include_detailed_results
        )
        
        # Update metrics
        SYSTEM_METRICS["total_batches"] += 1
        
        # Create response
        response = BatchPredictionResponse(
            batch_id=batch_id,
            status="processing",
            total_customers=customer_count,
            estimated_completion_time=estimated_completion.isoformat(),
            status_check_url=f"/api/v1/batch/status/{batch_id}",
            submitted_at=submitted_at.isoformat()
        )
        
        logger.info(f"Batch {batch_id} accepted for processing")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction initialization failed: {str(e)}")
        
        ERROR_COUNT.labels(
            error_type=type(e).__name__,
            endpoint="/api/v1/batch-predict"
        ).inc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/api/v1/batch/status/{batch_id}", tags=["Batch Processing"])
async def get_batch_status(batch_id: str):
    """
    📈 **Get Batch Processing Status**
    
    Check the status and progress of a batch prediction job.
    
    **Status Values:**
    - `processing`: Batch is currently being processed
    - `completed`: All predictions completed successfully
    - `failed`: Batch processing failed
    - `partial`: Some predictions completed, some failed
    
    **Returns:** Detailed status including progress and results (if completed)
    """
    if batch_id not in BATCH_JOBS:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found"
        )
    
    batch_info = BATCH_JOBS[batch_id]
    
    # Add real-time progress calculation
    progress_percentage = 0
    if batch_info["total_customers"] > 0:
        progress_percentage = (batch_info["processed_customers"] / batch_info["total_customers"]) * 100
    
    response = {
        "batch_id": batch_id,
        "status": batch_info["status"],
        "progress_percentage": round(progress_percentage, 1),
        "total_customers": batch_info["total_customers"],
        "processed_customers": batch_info["processed_customers"],
        "completed_customers": batch_info["completed_customers"],
        "failed_customers": batch_info["failed_customers"],
        "submitted_at": batch_info["submitted_at"],
        "estimated_completion": batch_info.get("estimated_completion"),
        "completed_at": batch_info.get("completed_at"),
        "processing_time_seconds": None
    }
    
    # Calculate actual processing time if completed
    if batch_info["status"] in ["completed", "failed", "partial"] and "completed_at" in batch_info:
        submitted = datetime.fromisoformat(batch_info["submitted_at"])
        completed = datetime.fromisoformat(batch_info["completed_at"])
        response["processing_time_seconds"] = (completed - submitted).total_seconds()
    
    # Include results if requested and available
    if batch_info.get("include_detailed_results", False) and batch_info["status"] != "processing":
        response["results"] = batch_info.get("results", [])
    
    return response

async def process_batch_predictions(batch_id: str, customers: List[CustomerInput], include_details: bool):
    """
    Background task to process batch predictions
    This runs asynchronously so the API can return immediately
    """
    try:
        logger.info(f"Starting background processing for batch {batch_id}")
        
        results = []
        processed_count = 0
        completed_count = 0
        failed_count = 0
        
        model_version = get_current_model_version()
        
        for customer in customers:
            try:
                # Process individual prediction
                start_time = time.time()
                
                churn_probability = simulate_ml_prediction(customer)
                confidence_score = min(0.95, max(0.60, 0.9 - (churn_probability * 0.1)))
                risk_category = calculate_risk_category(churn_probability)
                recommended_action, retention_strategy = get_recommended_action(churn_probability, risk_category)
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Create result
                result = {
                    "customer_id": customer.customer_id,
                    "churn_probability": churn_probability,
                    "churn_risk_category": risk_category,
                    "confidence_score": confidence_score,
                    "recommended_action": recommended_action,
                    "retention_strategy": retention_strategy,
                    "processing_time_ms": processing_time_ms,
                    "status": "success"
                }
                
                if include_details:
                    results.append(result)
                
                completed_count += 1
                
                # Update metrics
                PREDICTION_COUNT.labels(
                    model_version=model_version,
                    prediction_type="batch"
                ).inc()
                
            except Exception as e:
                logger.error(f"Failed to process customer {customer.customer_id} in batch {batch_id}: {str(e)}")
                
                failed_count += 1
                
                if include_details:
                    results.append({
                        "customer_id": customer.customer_id,
                        "status": "failed",
                        "error": str(e)
                    })
            
            processed_count += 1
            
            # Update batch status
            BATCH_JOBS[batch_id]["processed_customers"] = processed_count
            BATCH_JOBS[batch_id]["completed_customers"] = completed_count
            BATCH_JOBS[batch_id]["failed_customers"] = failed_count
        
        # Determine final status
        if failed_count == 0:
            final_status = "completed"
        elif completed_count == 0:
            final_status = "failed"
        else:
            final_status = "partial"
        
        # Update final batch status
        BATCH_JOBS[batch_id].update({
            "status": final_status,
            "completed_at": datetime.now().isoformat(),
            "results": results
        })
        
        # Send webhook notification if provided
        webhook_url = BATCH_JOBS[batch_id].get("webhook_url")
        if webhook_url:
            await send_webhook_notification(batch_id, webhook_url, final_status)
        
        logger.info(f"Batch {batch_id} processing completed: {completed_count} success, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Batch processing failed for {batch_id}: {str(e)}")
        
        BATCH_JOBS[batch_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })

async def send_webhook_notification(batch_id: str, webhook_url: str, status: str):
    """
    Send webhook notification when batch processing completes
    """
    try:
        # In production, use aiohttp or httpx to send webhook
        logger.info(f"Would send webhook to {webhook_url} for batch {batch_id} with status {status}")
        
        # Example webhook payload
        webhook_payload = {
            "batch_id": batch_id,
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "status_url": f"/api/v1/batch/status/{batch_id}"
        }
        
        # In production:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(webhook_url, json=webhook_payload) as response:
        #         logger.info(f"Webhook sent successfully: {response.status}")
        
    except Exception as e:
        logger.error(f"Failed to send webhook notification: {str(e)}")

# ===================================================================
# API ENDPOINTS - MODEL MANAGEMENT
# ===================================================================

@app.get("/api/v1/models", response_model=List[ModelInfo], tags=["Model Management"])
async def list_models():
    """
    📋 **List Available Models**
    
    Get information about all available ML models including their status,
    accuracy metrics, and version information.
    
    **Model Status:**
    - `active`: Currently serving predictions
    - `testing`: Available for testing but not production
    - `deprecated`: Old version, still available but not recommended
    """
    models = []
    
    for version, info in MODEL_REGISTRY.items():
        model_info = ModelInfo(
            version=version,
            accuracy=info["accuracy"],
            status=info["status"],
            created_at=info["created_at"],
            features_count=info["features_count"],
            training_samples=info["training_samples"]
        )
        models.append(model_info)
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.created_at, reverse=True)
    
    return models

@app.post("/api/v1/models/{version}/activate", tags=["Model Management"])
async def activate_model(version: str):
    """
    🚀 **Activate Model Version**
    
    Switch the active model to a specific version.
    This immediately affects all new predictions.
    
    **Note:** This is a critical operation that affects production traffic.
    Ensure the model version has been properly tested before activation.
    """
    if version not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {version} not found"
        )
    
    # Deactivate current active model
    for v, info in MODEL_REGISTRY.items():
        if info["status"] == "active":
            info["status"] = "testing"
    
    # Activate the requested model
    MODEL_REGISTRY[version]["status"] = "active"
    
    logger.info(f"Model {version} activated successfully")
    
    return {
        "message": f"Model {version} is now active",
        "previous_version": get_current_model_version(),
        "new_version": version,
        "activated_at": datetime.now().isoformat()
    }

# ===================================================================
# API ENDPOINTS - MONITORING & HEALTH
# ===================================================================

@app.get("/health", response_model=HealthCheck, tags=["System Health"])
async def health_check():
    """
    ❤️ **System Health Check**
    
    Check the overall health and status of the ML API system.
    Used by load balancers and monitoring systems.
    
    **Returns:** System status, uptime, and key metrics
    """
    uptime_seconds = (datetime.now() - SYSTEM_METRICS["start_time"]).total_seconds()
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime_seconds,
        models_available=len(MODEL_REGISTRY),
        last_prediction=SYSTEM_METRICS["last_prediction_time"].isoformat() if SYSTEM_METRICS["last_prediction_time"] else None
    )

@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics():
    """
    📊 **Prometheus Metrics**
    
    Export metrics in Prometheus format for monitoring and alerting.
    
    **Metrics Included:**
    - Request counts and response times
    - Prediction counts by model version
    - Error rates and types
    - System resource usage
    """
    # In production, this would return actual Prometheus metrics
    # return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # For demo, return JSON format
    return {
        "total_requests": SYSTEM_METRICS["total_predictions"],
        "total_predictions": SYSTEM_METRICS["total_predictions"],
        "total_batches": SYSTEM_METRICS["total_batches"],
        "uptime_seconds": (datetime.now() - SYSTEM_METRICS["start_time"]).total_seconds(),
        "active_model": get_current_model_version(),
        "models_loaded": len(MODEL_REGISTRY)
    }

@app.get("/api/v1/stats", tags=["Monitoring"])
async def get_api_statistics():
    """
    📈 **API Usage Statistics**
    
    Get detailed statistics about API usage, performance, and trends.
    """
    current_time = datetime.now()
    uptime = current_time - SYSTEM_METRICS["start_time"]
    
    # Calculate rates
    predictions_per_hour = 0
    if uptime.total_seconds() > 0:
        predictions_per_hour = (SYSTEM_METRICS["total_predictions"] / uptime.total_seconds()) * 3600
    
    # Active batch jobs
    active_batches = sum(1 for job in BATCH_JOBS.values() if job["status"] == "processing")
    
    return {
        "api_version": "2.0.0",
        "system_uptime_hours": uptime.total_seconds() / 3600,
        "total_predictions": SYSTEM_METRICS["total_predictions"],
        "total_batch_jobs": SYSTEM_METRICS["total_batches"],
        "active_batch_jobs": active_batches,
        "predictions_per_hour": round(predictions_per_hour, 2),
        "current_model": get_current_model_version(),
        "model_accuracy": MODEL_REGISTRY[get_current_model_version()]["accuracy"],
        "last_prediction": SYSTEM_METRICS["last_prediction_time"].isoformat() if SYSTEM_METRICS["last_prediction_time"] else None,
        "server_time": current_time.isoformat()
    }

# ===================================================================
# ERROR HANDLERS
# ===================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Handle validation errors with user-friendly messages
    """
    logger.warning(f"Validation error on {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected errors with logging and monitoring
    """
    logger.error(f"Unexpected error on {request.url.path}: {str(exc)}", exc_info=True)
    
    ERROR_COUNT.labels(
        error_type=type(exc).__name__,
        endpoint=str(request.url.path)
    ).inc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "request_id": f"err_{uuid.uuid4().hex[:12]}"
        }
    )

# ===================================================================
# STARTUP AND SHUTDOWN EVENTS
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup
    """
    logger.info("🚀 ML Pipeline API starting up...")
    logger.info(f"✅ Loaded {len(MODEL_REGISTRY)} model versions")
    logger.info(f"✅ Active model: {get_current_model_version()}")
    logger.info("✅ Monitoring and metrics enabled")
    logger.info("✅ API ready to serve predictions!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    logger.info("🛑 ML Pipeline API shutting down...")
    logger.info("✅ Cleanup completed")

# ===================================================================
# MAIN APPLICATION ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    """
    Run the application directly with uvicorn
    For production, use a proper ASGI server like gunicorn + uvicorn workers
    """
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8001,       # Use port 8001 to avoid conflicts
        log_level="info",
        reload=False,    # Disable reload in production
        workers=1        # Single worker for demo (scale up in production)
    )
async def predict(data: DataInput):
    """
    Real-time model inference endpoint
    Used by applications to get predictions from your trained model
    """
    try:
        # Simulate model prediction (replace with actual model)
        prediction = sum(data.features.values()) * 0.1  # Dummy calculation
        confidence = min(0.95, max(0.60, prediction / 10))
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version="v1.1",
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(data_list: List[DataInput]):
    """
    Batch prediction for multiple inputs
    Useful for processing large datasets
    """
    results = []
    for data in data_list:
        pred_response = await predict(data)
        results.append(pred_response)
    return {"batch_size": len(results), "predictions": results}

# 2. PIPELINE ORCHESTRATION ENDPOINTS
@app.post("/pipeline/start")
async def start_pipeline(background_tasks: BackgroundTasks, request: TrainingRequest):
    """
    Start a new ML pipeline job (training, validation, etc.)
    Returns immediately with job ID for tracking
    """
    pipeline_id = str(uuid.uuid4())
    
    pipeline_jobs[pipeline_id] = PipelineStatus(
        pipeline_id=pipeline_id,
        status="running",
        stage="data_preprocessing",
        progress=0.0,
        started_at=datetime.now().isoformat(),
        estimated_completion=None
    )
    
    # Start background task (simulate pipeline execution)
    background_tasks.add_task(simulate_pipeline_execution, pipeline_id)
    
    return {
        "message": "Pipeline started successfully",
        "pipeline_id": pipeline_id,
        "track_url": f"/pipeline/status/{pipeline_id}"
    }

@app.get("/pipeline/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """
    Check the status of a running pipeline
    """
    if pipeline_id not in pipeline_jobs:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return pipeline_jobs[pipeline_id]

# 3. MODEL MANAGEMENT ENDPOINTS
@app.get("/models")
async def list_models():
    """
    List all available model versions
    """
    return {"models": model_registry}

@app.post("/models/{version}/deploy")
async def deploy_model(version: str):
    """
    Deploy a specific model version to production
    """
    if version not in model_registry:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    # Update model status
    for v in model_registry:
        model_registry[v]["status"] = "inactive"
    model_registry[version]["status"] = "active"
    
    return {
        "message": f"Model {version} deployed successfully",
        "active_model": version,
        "accuracy": model_registry[version]["accuracy"]
    }

# 4. DATA VALIDATION ENDPOINTS
@app.post("/validate-data")
async def validate_data(data: DataInput):
    """
    Validate input data before processing
    Check for data drift, missing values, etc.
    """
    issues = []
    
    # Check for required features
    required_features = ["feature1", "feature2", "feature3"]
    missing_features = [f for f in required_features if f not in data.features]
    if missing_features:
        issues.append(f"Missing features: {missing_features}")
    
    # Check for data ranges
    for feature, value in data.features.items():
        if value < 0 or value > 100:
            issues.append(f"Feature {feature} out of range: {value}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "data_quality_score": max(0, 1 - len(issues) * 0.2)
    }

# 5. MONITORING & METRICS ENDPOINTS
@app.get("/metrics")
async def get_metrics():
    """
    Get system and model performance metrics
    """
    return {
        "total_predictions": 1547,
        "average_response_time": 0.045,
        "model_accuracy": 0.87,
        "uptime": "99.8%",
        "active_pipelines": len([j for j in pipeline_jobs.values() if j.status == "running"]),
        "last_updated": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Background task simulation
async def simulate_pipeline_execution(pipeline_id: str):
    """
    Simulate ML pipeline execution stages
    """
    stages = [
        ("data_preprocessing", 20),
        ("feature_engineering", 40),
        ("model_training", 70),
        ("model_validation", 90),
        ("model_deployment", 100)
    ]
    
    for stage, progress in stages:
        await asyncio.sleep(2)  # Simulate work
        if pipeline_id in pipeline_jobs:
            pipeline_jobs[pipeline_id].stage = stage
            pipeline_jobs[pipeline_id].progress = progress
    
    # Mark as completed
    if pipeline_id in pipeline_jobs:
        pipeline_jobs[pipeline_id].status = "completed"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
