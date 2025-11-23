#!/usr/bin/env python3
"""
Script para probar la API de clustering cuando esté funcionando
"""
import requests
import json

# URL de la API (cuando Docker esté corriendo)
API_URL = "http://localhost:8001"

def test_api():
    """Función para probar la API"""
    try:
        # Probar el endpoint raíz
        print("Probando endpoint raíz...")
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
        
        # Probar el endpoint de predicción
        print("Probando endpoint de predicción...")
        
        # Datos de ejemplo (zona típica de NYC)
        sample_data = {
            "total_trips": 1500.0,
            "avg_pickup_hour": 14.5,
            "std_pickup_hour": 6.2,
            "peak_hour": 18.0,
            "peak_hour_concentration_pct": 8.5,
            "pct_AM_Peak": 15.2,
            "pct_PM_Peak": 18.7,
            "pct_OP_day": 45.3,
            "pct_OP_night": 20.8,
            "pct_Early_Morning": 5.0,
            "pct_weekend_trips": 22.1,
            "avg_trip_distance": 2.8,
            "median_trip_distance": 1.9,
            "avg_trip_duration": 12.5,
            "avg_fare": 15.50,
            "avg_total_amount": 19.25,
            "avg_tip": 2.75,
            "avg_tolls": 0.85,
            "pct_yellow_taxi": 65.5
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=sample_data
        )
        
        print(f"Status: {response.status_code}")
        print(f"Prediction: {response.json()}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("No se puede conectar a la API. Asegúrate de que Docker esté corriendo.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Probando la API de Clustering de Zonas de NYC")
    print("=" * 50)
    test_api()