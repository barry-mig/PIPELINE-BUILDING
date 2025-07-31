from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, List, Optional
import random
import json
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import asyncio

# The "Straight Arrow" API for A/B Testing ML Models
app = FastAPI(
    title="ML A/B Testing API",
    description="Lightning-fast API for testing multiple ML models in production",
    version="2.0.0"
)

# Data Models
class PredictionRequest(BaseModel):
    features: Dict[str, float]
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    experiment_id: str
    timestamp: str
    response_time_ms: float

class ExperimentConfig(BaseModel):
    name: str
    models: Dict[str, float]  # model_version: traffic_percentage
    target_metric: str
    duration_days: int

class ModelPerformance(BaseModel):
    model_version: str
    predictions_count: int
    avg_confidence: float
    avg_response_time: float
    success_rate: float
    business_metric: float

# In-memory storage (production would use Redis/Database)
experiments = {}
model_performance = defaultdict(lambda: {
    "predictions": 0,
    "total_confidence": 0,
    "total_response_time": 0,
    "successes": 0,
    "business_value": 0
})

# Mock ML Models (replace with real models)
class MockMLModel:
    def __init__(self, version: str, accuracy: float, latency_ms: float):
        self.version = version
        self.accuracy = accuracy
        self.latency_ms = latency_ms
    
    async def predict(self, features: Dict[str, float]):
        # Simulate model latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Simple prediction logic
        prediction = sum(features.values()) * 0.1 * (1 + self.accuracy)
        confidence = min(0.95, max(0.60, self.accuracy + random.uniform(-0.1, 0.1)))
        
        return prediction, confidence

# Available Models
models = {
    "v1.0": MockMLModel("v1.0", 0.85, 45),  # Current prod model
    "v2.0": MockMLModel("v2.0", 0.89, 60),  # New model (higher accuracy, slower)
    "v2.1": MockMLModel("v2.1", 0.87, 30),  # Optimized model (balanced)
    "experimental": MockMLModel("experimental", 0.91, 80)  # Bleeding edge
}

# Initialize default experiment
experiments["default"] = {
    "name": "Production A/B Test",
    "models": {"v1.0": 0.8, "v2.0": 0.2},  # 80% v1.0, 20% v2.0
    "target_metric": "business_value",
    "duration_days": 7,
    "started_at": datetime.now(),
    "status": "active"
}

# 🚀 CORE A/B TESTING ENDPOINTS

@app.post("/predict", response_model=PredictionResponse)
async def smart_predict(
    request: PredictionRequest,
    experiment_id: str = "default",
    user_agent: str = Header(None)
):
    """
    THE STRAIGHT ARROW 🏹
    Instantly route requests to the best model based on A/B test configuration
    """
    start_time = datetime.now()
    
    # Get experiment configuration
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments[experiment_id]
    
    # Select model based on traffic allocation
    selected_model = select_model_for_user(
        experiment["models"], 
        request.user_id or str(uuid.uuid4())
    )
    
    # Get prediction from selected model
    if selected_model not in models:
        raise HTTPException(status_code=404, detail=f"Model {selected_model} not found")
    
    model = models[selected_model]
    prediction, confidence = await model.predict(request.features)
    
    # Calculate response time
    response_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Track performance metrics
    track_model_performance(selected_model, confidence, response_time, True)
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=selected_model,
        experiment_id=experiment_id,
        timestamp=start_time.isoformat(),
        response_time_ms=response_time
    )

@app.post("/experiments")
async def create_experiment(config: ExperimentConfig):
    """
    Create a new A/B testing experiment
    """
    # Validate traffic percentages sum to 1.0
    total_traffic = sum(config.models.values())
    if abs(total_traffic - 1.0) > 0.01:
        raise HTTPException(
            status_code=400, 
            detail=f"Traffic percentages must sum to 1.0, got {total_traffic}"
        )
    
    experiment_id = str(uuid.uuid4())
    experiments[experiment_id] = {
        "name": config.name,
        "models": config.models,
        "target_metric": config.target_metric,
        "duration_days": config.duration_days,
        "started_at": datetime.now(),
        "status": "active"
    }
    
    return {
        "experiment_id": experiment_id,
        "message": "Experiment created and activated",
        "config": config
    }

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """
    Get real-time A/B test results
    """
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments[experiment_id]
    results = []
    
    for model_version in experiment["models"]:
        perf = model_performance[model_version]
        if perf["predictions"] > 0:
            results.append(ModelPerformance(
                model_version=model_version,
                predictions_count=perf["predictions"],
                avg_confidence=perf["total_confidence"] / perf["predictions"],
                avg_response_time=perf["total_response_time"] / perf["predictions"],
                success_rate=perf["successes"] / perf["predictions"],
                business_metric=perf["business_value"] / perf["predictions"]
            ))
    
    # Calculate statistical significance
    winner = determine_winner(results) if len(results) >= 2 else None
    
    return {
        "experiment": experiment,
        "results": results,
        "winner": winner,
        "recommendation": get_recommendation(results)
    }

# 🎯 INSTANT MODEL SWITCHING

@app.post("/switch-traffic")
async def switch_traffic(experiment_id: str, new_allocation: Dict[str, float]):
    """
    Instantly change traffic allocation - THE POWER OF APIs!
    No downtime, no deployment, just change and go!
    """
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Validate allocation
    if abs(sum(new_allocation.values()) - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Traffic must sum to 1.0")
    
    old_allocation = experiments[experiment_id]["models"]
    experiments[experiment_id]["models"] = new_allocation
    
    return {
        "message": "Traffic switched instantly! 🚀",
        "old_allocation": old_allocation,
        "new_allocation": new_allocation,
        "switched_at": datetime.now().isoformat()
    }

@app.post("/emergency-rollback/{model_version}")
async def emergency_rollback(model_version: str):
    """
    EMERGENCY! Instantly route 100% traffic to a specific model
    """
    if model_version not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Update all active experiments
    for exp_id in experiments:
        experiments[exp_id]["models"] = {model_version: 1.0}
    
    return {
        "message": f"🚨 EMERGENCY ROLLBACK COMPLETE! 🚨",
        "all_traffic_routed_to": model_version,
        "timestamp": datetime.now().isoformat()
    }

# 📊 REAL-TIME MONITORING

@app.get("/dashboard")
async def get_dashboard():
    """
    Real-time dashboard data
    """
    total_predictions = sum(perf["predictions"] for perf in model_performance.values())
    
    model_stats = {}
    for model_id, perf in model_performance.items():
        if perf["predictions"] > 0:
            model_stats[model_id] = {
                "predictions": perf["predictions"],
                "avg_response_time": perf["total_response_time"] / perf["predictions"],
                "avg_confidence": perf["total_confidence"] / perf["predictions"],
                "success_rate": perf["successes"] / perf["predictions"],
                "traffic_share": perf["predictions"] / total_predictions if total_predictions > 0 else 0
            }
    
    return {
        "total_predictions": total_predictions,
        "active_experiments": len([e for e in experiments.values() if e["status"] == "active"]),
        "model_performance": model_stats,
        "last_updated": datetime.now().isoformat()
    }

# 🔧 UTILITY FUNCTIONS

def select_model_for_user(model_allocation: Dict[str, float], user_id: str) -> str:
    """
    Consistent model selection based on user ID
    Same user always gets same model (important for user experience)
    """
    # Use hash of user_id for consistent assignment
    user_hash = hash(user_id) % 100
    
    cumulative = 0
    for model, percentage in model_allocation.items():
        cumulative += percentage * 100
        if user_hash < cumulative:
            return model
    
    # Fallback to first model
    return list(model_allocation.keys())[0]

def track_model_performance(model_version: str, confidence: float, response_time: float, success: bool):
    """Track model performance metrics"""
    perf = model_performance[model_version]
    perf["predictions"] += 1
    perf["total_confidence"] += confidence
    perf["total_response_time"] += response_time
    if success:
        perf["successes"] += 1
    # Simulate business value (would be real metrics in production)
    perf["business_value"] += confidence * random.uniform(0.8, 1.2)

def determine_winner(results: List[ModelPerformance]) -> Optional[str]:
    """Simple winner determination (in production, use proper statistical tests)"""
    if len(results) < 2:
        return None
    
    # Find model with highest business metric and sufficient sample size
    best = max(results, key=lambda x: x.business_metric if x.predictions_count > 100 else 0)
    
    if best.predictions_count > 100:
        return best.model_version
    return None

def get_recommendation(results: List[ModelPerformance]) -> str:
    """Get recommendation based on results"""
    if not results:
        return "Not enough data yet"
    
    best = max(results, key=lambda x: x.business_metric)
    
    if best.predictions_count < 100:
        return "Continue testing - need more data"
    elif best.business_metric > 0.8:
        return f"Winner: {best.model_version} - recommend 100% traffic"
    else:
        return "Results inconclusive - continue testing"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
