from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="NYC Taxi Zone Clustering API", description="API for predicting cluster assignment of NYC Taxi Zones", version="1.0.0")

# Load model and scaler
MODEL_PATH = "app/models_assets/kmeans_model.pkl"
SCALER_PATH = "app/models_assets/feature_scaler.pkl"

model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        # In production, you might want to raise an error or exit
        pass

class ZoneFeatures(BaseModel):
    total_trips: float
    avg_pickup_hour: float
    std_pickup_hour: float
    peak_hour: float
    peak_hour_concentration_pct: float
    pct_AM_Peak: float
    pct_PM_Peak: float
    pct_OP_day: float
    pct_OP_night: float
    pct_Early_Morning: float
    pct_weekend_trips: float
    avg_trip_distance: float
    median_trip_distance: float
    avg_trip_duration: float
    avg_fare: float
    avg_total_amount: float
    avg_tip: float
    avg_tolls: float
    pct_yellow_taxi: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the NYC Taxi Zone Clustering API. Use /predict to get cluster assignments."}

@app.post("/predict")
def predict_cluster(features: ZoneFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")
    
    try:
        # Convert features to dataframe/array in the correct order
        data = {
            "total_trips": [features.total_trips],
            "avg_pickup_hour": [features.avg_pickup_hour],
            "std_pickup_hour": [features.std_pickup_hour],
            "peak_hour": [features.peak_hour],
            "peak_hour_concentration_pct": [features.peak_hour_concentration_pct],
            "pct_AM_Peak": [features.pct_AM_Peak],
            "pct_PM_Peak": [features.pct_PM_Peak],
            "pct_OP_day": [features.pct_OP_day],
            "pct_OP_night": [features.pct_OP_night],
            "pct_Early_Morning": [features.pct_Early_Morning],
            "pct_weekend_trips": [features.pct_weekend_trips],
            "avg_trip_distance": [features.avg_trip_distance],
            "median_trip_distance": [features.median_trip_distance],
            "avg_trip_duration": [features.avg_trip_duration],
            "avg_fare": [features.avg_fare],
            "avg_total_amount": [features.avg_total_amount],
            "avg_tip": [features.avg_tip],
            "avg_tolls": [features.avg_tolls],
            "pct_yellow_taxi": [features.pct_yellow_taxi]
        }
        
        df = pd.DataFrame(data)
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Predict
        cluster = model.predict(scaled_features)[0]
        
        return {
            "cluster": int(cluster),
            "features_received": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
