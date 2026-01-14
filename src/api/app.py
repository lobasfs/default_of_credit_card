"""
FastAPI application for model serving
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib

app = FastAPI(
    title="Credit Card Default Prediction API",
    description="API for predicting credit card default probability",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    """Input schema for prediction"""
    LIMIT_BAL: float = Field(..., description="Amount of given credit")
    SEX: int = Field(..., ge=1, le=2, description="Gender (1=male, 2=female)")
    EDUCATION: int = Field(..., ge=0, le=6, description="Education level")
    MARRIAGE: int = Field(..., ge=0, le=3, description="Marital status")
    AGE: int = Field(..., ge=18, le=100, description="Age in years")
    PAY_0: int = Field(..., ge=-2, le=8, description="Repayment status September")
    PAY_2: int = Field(..., ge=-2, le=8, description="Repayment status August")
    PAY_3: int = Field(..., ge=-2, le=8, description="Repayment status July")
    PAY_4: int = Field(..., ge=-2, le=8, description="Repayment status June")
    PAY_5: int = Field(..., ge=-2, le=8, description="Repayment status May")
    PAY_6: int = Field(..., ge=-2, le=8, description="Repayment status April")
    BILL_AMT1: float = Field(..., description="Bill amount September")
    BILL_AMT2: float = Field(..., description="Bill amount August")
    BILL_AMT3: float = Field(..., description="Bill amount July")
    BILL_AMT4: float = Field(..., description="Bill amount June")
    BILL_AMT5: float = Field(..., description="Bill amount May")
    BILL_AMT6: float = Field(..., description="Bill amount April")
    PAY_AMT1: float = Field(..., ge=0, description="Payment amount September")
    PAY_AMT2: float = Field(..., ge=0, description="Payment amount August")
    PAY_AMT3: float = Field(..., ge=0, description="Payment amount July")
    PAY_AMT4: float = Field(..., ge=0, description="Payment amount June")
    PAY_AMT5: float = Field(..., ge=0, description="Payment amount May")
    PAY_AMT6: float = Field(..., ge=0, description="Payment amount April")
    
    class Config:
        schema_extra = {
            "example": {
                "LIMIT_BAL": 20000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913.0,
                "BILL_AMT2": 3102.0,
                "BILL_AMT3": 689.0,
                "BILL_AMT4": 0.0,
                "BILL_AMT5": 0.0,
                "BILL_AMT6": 0.0,
                "PAY_AMT1": 0.0,
                "PAY_AMT2": 689.0,
                "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0,
                "PAY_AMT5": 0.0,
                "PAY_AMT6": 0.0
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: int = Field(..., description="Predicted class (0=no default, 1=default)")
    probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")


# Global variable to store model
model = None


def load_model(model_path: str = "models/model.pkl"):
    """Load trained model"""
    global model
    
    try:
        # Try to load from file
        if Path(model_path).exists():
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            # Try to load from MLflow
            model_uri = "models:/credit_card_default/latest"
            try:
                model = mlflow.sklearn.load_model(model_uri)
                print(f"Model loaded from MLflow: {model_uri}")
            except Exception as e:
                print(f"Could not load model from MLflow: {e}")
                raise FileNotFoundError(f"Model not found at {model_path} or in MLflow")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to input data"""
    from src.features.engineer import engineer_features as eng_feat
    return eng_feat(df)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will need to be loaded manually or predictions will fail")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Card Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make prediction for a single instance
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(input_data: List[PredictionInput]):
    """
    Make predictions for multiple instances
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert inputs to DataFrame
        input_dicts = [item.dict() for item in input_data]
        df = pd.DataFrame(input_dicts)
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Prepare results
        results = []
        for pred, prob in zip(predictions, probabilities):
            if prob < 0.3:
                risk_level = "low"
            elif prob < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            results.append({
                "prediction": int(pred),
                "probability": float(prob),
                "risk_level": risk_level
            })
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
