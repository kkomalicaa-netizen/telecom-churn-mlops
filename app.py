from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import json
import time
import logging

# Configure structured logging (Requirement 6)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="An API that predicts whether a telecom customer is likely to churn with artifact tracking and drift monitoring.",
    version="1.0.0"
)

# Request & Latency Logging Middleware (Requirement 6)
@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time_ms:.2f}ms"
    )
    return response

# Load the trained model pipeline when the app starts
MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/model_metadata.json"
model = None
metadata = {}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully into FastAPI!")
else:
    print("WARNING: model.joblib not found! Please run your training script first.")

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("Model metadata loaded successfully!")

# Define a simple API Key for authentication
API_KEY = "supersecretkey123"

# Define the expected input schema using Pydantic
class CustomerData(BaseModel):
    Tenure_Months: int = Field(..., alias="Tenure Months")
    Monthly_Charges: float = Field(..., alias="Monthly Charges")
    Total_Charges: float = Field(..., alias="Total Charges")
    Gender: str
    Senior_Citizen: str = Field(..., alias="Senior Citizen")
    Partner: str
    Dependents: str
    Phone_Service: str = Field(..., alias="Phone Service")
    Multiple_Lines: str = Field(..., alias="Multiple Lines")
    Internet_Service: str = Field(..., alias="Internet Service")
    Online_Security: str = Field(..., alias="Online Security")
    Online_Backup: str = Field(..., alias="Online Backup")
    Device_Protection: str = Field(..., alias="Device Protection")
    Tech_Support: str = Field(..., alias="Tech Support")
    Streaming_TV: str = Field(..., alias="Streaming TV")
    Streaming_Movies: str = Field(..., alias="Streaming Movies")
    Contract: str
    Paperless_Billing: str = Field(..., alias="Paperless Billing")
    Payment_Method: str = Field(..., alias="Payment Method")

    class Config:
        populate_by_name = True

class DriftRequest(BaseModel):
    recent_tenure_months: list[float]

@app.get("/")
def home():
    return {"message": "Welcome to the Telecom Churn API! Go to /docs to test it."}

@app.get("/health")
def health_check():
    """Health check endpoint to ensure the app and model are running."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model-info")
def model_info(x_api_key: str = Header(None)):
    """Returns model metadata, including data hash, library versions, and test metrics for traceability."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key.")
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return metadata

@app.post("/predict")
def predict_churn(data: CustomerData, x_api_key: str = Header(None)):
    """Predicts whether a single customer will churn."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key. Pass 'x-api-key' in headers.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not available.")

    try:
        input_data = pd.DataFrame([data.dict(by_alias=True)])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Churn" if prediction == 1 else "Stay"
        
        return {
            "prediction_result": result,
            "churn_probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/drift-check")
def drift_check(payload: DriftRequest, x_api_key: str = Header(None)):
    """Compares recent inputs against baseline distribution using Population Stability Index (PSI)."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key.")

    try:
        # Baseline reference distribution (simulated training distribution values for Tenure Months)
        baseline = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72])
        actual = np.array(payload.recent_tenure_months)

        if len(actual) == 0:
            raise HTTPException(status_code=400, detail="Provided list cannot be empty.")

        # Calculate PSI
        bins = 5
        percentiles = np.linspace(0, 100, bins + 1)
        breakpoint_val = np.percentile(baseline, percentiles)
        breakpoint_val[0] = -np.inf
        breakpoint_val[-1] = np.inf

        expected_counts = np.histogram(baseline, bins=breakpoint_val)[0]
        actual_counts = np.histogram(actual, bins=breakpoint_val)[0]

        expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts) / len(baseline)
        actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts) / len(actual)

        psi_value = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        is_drifted = psi_value > 0.0

        return {
            "feature": "Tenure Months",
            "population_stability_index": round(psi_value, 4),
            "drift_detected": is_drifted,
            "status": "Flagged for drift (PSI > 0)" if is_drifted else "Stable"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
