#!/usr/bin/env python3
"""
Simple script to run the FastAPI app locally for testing
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting NYC Taxi Zone Clustering API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host="localhost",
        port=8000,
        reload=True
    )