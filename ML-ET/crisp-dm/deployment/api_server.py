import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

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
    avg_trip_distance = features_dict.get('avg_trip_distance', 0)
    
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
    status = "DEMO MODE" if DEMO_MODE else "PRODUCTION"
    return {
        "message": "Welcome to the NYC Taxi Zone Clustering API",
        "status": status,
        "endpoints": {
            "/": "This welcome message",
            "/predict": "POST - Predict cluster for zone features",
            "/health": "Health check",
            "/demo": "Demo cluster assignments",
            "/docs": "API Documentation (Swagger UI)"
        },
        "version": "1.0.0",
        "demo_mode": DEMO_MODE
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "demo_mode": DEMO_MODE,
        "timestamp": str(pd.Timestamp.now()),
        "api_version": "1.0.0"
    }

@app.get("/demo")
def get_demo_clusters():
    """Retorna ejemplos de clusters demo"""
    return {
        "cluster_descriptions": {
            "0": "Alta demanda, tarifas altas - Zonas comerciales premium (Times Square, Financial District)",
            "1": "Demanda media, zona entretenimiento - Áreas recreativas (Greenwich Village, SoHo)",
            "2": "Tarifas bajas, viajes cortos - Zonas residenciales (Queens, Brooklyn)",
            "3": "Baja demanda - Áreas periféricas (Staten Island, Bronx outer)", 
            "4": "Demanda estándar - Zonas mixtas (Midtown, Upper East/West Side)"
        },
        "total_clusters": 5,
        "clustering_algorithm": "K-Means (demo rules)",
        "features_used": [
            "total_trips", "avg_fare", "pct_weekend_trips", 
            "avg_trip_distance", "peak_hour_concentration"
        ]
    }

@app.post("/predict")
def predict_cluster(features: ZoneFeatures):
    try:
        if DEMO_MODE:
            # Modo demo con reglas simples
            cluster = demo_predict_cluster(features.dict())
            
            cluster_info = {
                0: "Alta demanda, tarifas altas",
                1: "Demanda media, zona entretenimiento", 
                2: "Tarifas bajas, viajes cortos",
                3: "Baja demanda",
                4: "Demanda estándar"
            }
            
            # Calcular score de confianza simulado
            confidence = min(0.95, 0.7 + (features.total_trips / 5000) * 0.2)
            
            return {
                "cluster": int(cluster),
                "cluster_description": cluster_info.get(cluster, "Cluster desconocido"),
                "confidence_score": round(confidence, 3),
                "mode": "demo",
                "features_summary": {
                    "total_trips": features.total_trips,
                    "avg_fare": f"${features.avg_fare:.2f}",
                    "avg_trip_distance": f"{features.avg_trip_distance:.1f} millas",
                    "pct_weekend_trips": f"{features.pct_weekend_trips:.1f}%"
                },
                "recommendations": get_cluster_recommendations(cluster)
            }
        else:
            # Aquí iría la predicción real con modelos cargados
            raise HTTPException(status_code=501, detail="Production mode not yet implemented")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

def get_cluster_recommendations(cluster):
    """Obtiene recomendaciones basadas en el cluster"""
    recommendations = {
        0: ["Aumentar flota durante horas pico", "Monitoreo en tiempo real", "Tarifas premium"],
        1: ["Servicio nocturno extendido", "Promociones weekend", "Rutas hacia aeropuertos"],
        2: ["Optimizar rutas cortas", "Servicios compartidos", "Tarifas competitivas"],
        3: ["Servicio bajo demanda", "Pre-reservas", "Cobertura básica"],
        4: ["Distribución estándar", "Monitoreo regular", "Ajustes según demanda"]
    }
    return recommendations.get(cluster, ["Análisis adicional requerido"])

@app.get("/clusters/analysis")  
def get_cluster_analysis():
    """Análisis general de clusters"""
    return {
        "total_clusters": 5,
        "distribution_expected": {
            "cluster_0": "15% - Zonas premium",
            "cluster_1": "20% - Zonas entretenimiento", 
            "cluster_2": "30% - Zonas residenciales",
            "cluster_3": "25% - Zonas periféricas",
            "cluster_4": "10% - Zonas mixtas"
        },
        "key_metrics": [
            "total_trips", "avg_fare", "avg_trip_distance", 
            "pct_weekend_trips", "peak_hour_concentration_pct"
        ]
    }

if __name__ == "__main__":
    print("Iniciando API de Clustering de Zonas de NYC...")
    print("Modo: DEMO")
    print("URL: http://localhost:8001")
    print("Docs: http://localhost:8001/docs")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)