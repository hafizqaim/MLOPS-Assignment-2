from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variable to store the loaded model
model = None
MODEL_PATH = "models/model_airflow.pkl"


def load_model():
    """Load the trained model from disk"""
    global model
    try:
        # Try different possible paths
        paths_to_try = [
            MODEL_PATH,
            "../models/model_airflow.pkl",
            "/app/models/model_airflow.pkl",
            "models/model.pkl"  # Fallback to original model
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✓ Model loaded successfully from {path}")
                return True
        
        print("✗ Model file not found in any expected location")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    success = load_model()
    if not success:
        print("WARNING: Model not loaded. /predict endpoint will not work.")


class CustomerFeatures(BaseModel):
    """Input schema for customer features"""
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    income: float = Field(..., ge=0, description="Annual income in dollars")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    account_balance: float = Field(..., description="Account balance in dollars")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    num_products: int = Field(..., ge=1, le=10, description="Number of products (1-10)")
    has_credit_card: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    is_active_member: int = Field(..., ge=0, le=1, description="Active member (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 75000.0,
                "credit_score": 720,
                "account_balance": 15000.0,
                "tenure_months": 24,
                "num_products": 2,
                "has_credit_card": 1,
                "is_active_member": 1
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    churn_prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    churn_probability: float = Field(..., description="Probability of churn (0.0-1.0)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Model confidence percentage")


class HealthResponse(BaseModel):
    """Output schema for health check"""
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_path: str = Field(..., description="Path to model file")
    api_version: str = Field(..., description="API version")


@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MLOps Churn Prediction API",
        "student_id": "f223142",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check():
    """
    Check API health status and model availability
    
    Returns:
        HealthResponse: API health status and model information
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        api_version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict customer churn")
async def predict_churn(features: CustomerFeatures):
    """
    Predict whether a customer will churn based on their features
    
    Args:
        features: Customer features for prediction
        
    Returns:
        PredictionResponse: Churn prediction with probability
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model file exists."
        )
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            'age': features.age,
            'income': features.income,
            'credit_score': features.credit_score,
            'account_balance': features.account_balance,
            'tenure_months': features.tenure_months,
            'num_products': features.num_products,
            'has_credit_card': features.has_credit_card,
            'is_active_member': features.is_active_member
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get probability for the predicted class
        churn_probability = float(prediction_proba[1])  # Probability of churn (class 1)
        confidence = float(max(prediction_proba)) * 100
        
        # Create response
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(churn_probability, 4),
            prediction_label="Will Churn" if prediction == 1 else "Will Not Churn",
            confidence=round(confidence, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", summary="Batch prediction endpoint")
async def predict_batch(customers: List[CustomerFeatures]):
    """
    Predict churn for multiple customers at once
    
    Args:
        customers: List of customer features
        
    Returns:
        List of predictions for each customer
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        predictions = []
        for customer in customers:
            result = await predict_churn(customer)
            predictions.append(result)
        
        return {
            "total_customers": len(customers),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting MLOps Churn Prediction API...")
    print(f"Student ID: f223142")
    uvicorn.run(app, host="0.0.0.0", port=8000)
