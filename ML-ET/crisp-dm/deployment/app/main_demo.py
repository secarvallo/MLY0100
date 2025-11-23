from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json

app = FastAPI(
    title="NYC Taxi Zone Clustering API", 
    description="API for predicting cluster assignment of NYC Taxi Zones", 
    version="1.0.0"
)

# Simulación temporal mientras resolvemos la carga de modelos
DEMO_MODE = True  # Cambiará a False cuando los modelos funcionen

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

def demo_predict_cluster(features_dict):
    """
    Función de predicción demo basada en reglas simples
    mientras resolvemos la carga de modelos
    """
    # Reglas simples de clustering basadas en características
    total_trips = features_dict.get('total_trips', 0)
    avg_fare = features_dict.get('avg_fare', 0)
    pct_weekend_trips = features_dict.get('pct_weekend_trips', 0)
    
    if total_trips > 2000 and avg_fare > 20:
        return 0  # Cluster de alta demanda, tarifas altas
    elif total_trips > 1000 and pct_weekend_trips > 25:
        return 1  # Cluster de demanda media, zona de entretenimiento
    elif avg_fare < 12:
        return 2  # Cluster de tarifas bajas, viajes cortos
    elif total_trips < 500:
        return 3  # Cluster de baja demanda
    else:
        return 4  # Cluster estándar

@app.get("/")
def read_root():
    status = "🚧 DEMO MODE" if DEMO_MODE else "✅ PRODUCTION"
    return {
        "message": "Welcome to the NYC Taxi Zone Clustering API",
        "status": status,
        "endpoints": {
            "/": "This welcome message",
            "/predict": "POST - Predict cluster for zone features",
            "/health": "Health check",
            "/demo": "Demo cluster assignments"
        },
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "demo_mode": DEMO_MODE,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/demo")
def get_demo_clusters():
    """Retorna ejemplos de clusters demo"""
    return {
        "cluster_descriptions": {
            "0": "🔴 Alta demanda, tarifas altas - Zonas comerciales premium",
            "1": "🟡 Demanda media, zona entretenimiento - Áreas recreativas",
            "2": "🟢 Tarifas bajas, viajes cortos - Zonas residenciales",
            "3": "🔵 Baja demanda - Áreas periféricas", 
            "4": "⚪ Demanda estándar - Zonas mixtas"
        },
        "total_clusters": 5
    }

@app.post("/predict")
def predict_cluster(features: ZoneFeatures):
    try:
        if DEMO_MODE:
            # Modo demo con reglas simples
            cluster = demo_predict_cluster(features.dict())
            
            cluster_info = {
                0: "🔴 Alta demanda, tarifas altas",
                1: "🟡 Demanda media, zona entretenimiento", 
                2: "🟢 Tarifas bajas, viajes cortos",
                3: "🔵 Baja demanda",
                4: "⚪ Demanda estándar"
            }
            
            return {
                "cluster": int(cluster),
                "cluster_description": cluster_info.get(cluster, "Cluster desconocido"),
                "mode": "demo",
                "confidence": "simulated",
                "features_summary": {
                    "total_trips": features.total_trips,
                    "avg_fare": features.avg_fare,
                    "avg_trip_distance": features.avg_trip_distance,
                    "pct_weekend_trips": features.pct_weekend_trips
                }
            }
        else:
            # Aquí iría la predicción real con modelos cargados
            raise HTTPException(status_code=501, detail="Production mode not yet implemented")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")  
def predict_batch(features_list: list[ZoneFeatures]):
    """Predicción en lote para múltiples zonas"""
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 predicciones por lote")
    
    results = []
    for i, features in enumerate(features_list):
        try:
            prediction = predict_cluster(features)
            results.append({
                "zone_index": i,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "zone_index": i,
                "error": str(e)
            })
    
    return {
        "batch_size": len(features_list),
        "predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)