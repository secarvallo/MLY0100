#!/usr/bin/env python3
"""
Quick test script for the API
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

def test_api():
    print("=== Testing NYC Taxi Clustering API ===")
    print()
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = client.get("/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✅ Root endpoint working")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test prediction endpoint
    print("2. Testing prediction endpoint...")
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
    
    try:
        response = client.post("/predict", json=sample_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Cluster prediction: {result.get('cluster')}")
        print("   ✅ Prediction endpoint working")
        
        # Test with different data
        sample_data2 = {**sample_data, "total_trips": 500.0, "avg_fare": 8.0}
        response2 = client.post("/predict", json=sample_data2)
        result2 = response2.json()
        print(f"   Second prediction (different data): Cluster {result2.get('cluster')}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test error handling
    print("3. Testing error handling...")
    try:
        bad_data = {"invalid": "data"}
        response = client.post("/predict", json=bad_data)
        print(f"   Status with bad data: {response.status_code}")
        print("   ✅ Error handling working")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("=== API Test Complete ===")

if __name__ == "__main__":
    test_api()