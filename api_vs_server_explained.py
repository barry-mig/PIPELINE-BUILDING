# API vs Server - Simple Example

"""
API = The Contract (What you can do)
Server = The Implementation (How it's done)

Think of it like a restaurant:
- API = Menu (tells you what you can order and how)
- Server = Kitchen (actually makes the food)
"""

# This is an API SPECIFICATION (the contract)
class MLPipelineAPI:
    """
    API Contract for ML Pipeline
    
    Available Endpoints:
    - POST /predict: Send data, get prediction
    - GET /health: Check if system is working
    - POST /retrain: Start model retraining
    """
    
    def predict(self, input_data):
        """
        Input: {"features": {"age": 25, "income": 50000}}
        Output: {"prediction": 0.85, "model": "v2.0"}
        """
        pass
    
    def health_check(self):
        """
        Output: {"status": "healthy", "uptime": "99.9%"}
        """
        pass

# This is the SERVER IMPLEMENTATION (the actual code)
from fastapi import FastAPI
import joblib

app = FastAPI()  # This creates the SERVER

# Load your trained ML model
model = joblib.load("customer_churn_model.pkl")

# SERVER implements the API contract
@app.post("/predict")  # API endpoint
def predict(input_data: dict):  # Server function
    """
    This SERVER FUNCTION implements the API CONTRACT
    """
    # Server processes the request
    features = input_data["features"]
    prediction = model.predict([list(features.values())])[0]
    
    # Server returns response according to API contract
    return {
        "prediction": float(prediction),
        "model": "v2.0",
        "confidence": 0.85
    }

@app.get("/health")  # API endpoint
def health_check():  # Server function
    return {"status": "healthy", "uptime": "99.9%"}

# WHY THIS IS POWERFUL FOR ML PIPELINES:

"""
SCENARIO: E-commerce company wants to predict customer churn

WITHOUT APIs (Old Way):
1. Data scientist runs model manually
2. Saves results to CSV file
3. Engineer uploads CSV to database
4. Website reads from database (stale data)
5. Takes hours to update
6. Can't handle real-time requests

WITH APIs (New Way):
1. Website calls API: POST /predict {"customer_id": 123}
2. API instantly returns: {"churn_risk": 0.85, "recommend_discount": true}
3. Website immediately shows discount offer
4. Model can be updated without touching website code
5. Handles 1000s of simultaneous requests
6. A/B tests different models automatically

THE STRAIGHT ARROW EFFECT:
- SPEED: Millisecond responses vs hours
- SCALE: Handle thousands of users vs one at a time  
- RELIABILITY: Auto-recovery vs manual fixes
- FLEXIBILITY: Update models independently vs rebuild everything
"""

if __name__ == "__main__":
    import uvicorn
    # This RUNS THE SERVER that implements the API
    uvicorn.run(app, host="0.0.0.0", port=8000)
