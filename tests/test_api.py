import os
os.environ["TESTING"] = "True"

from fastapi.testclient import TestClient
import pytest
from src.api.main import app

client = TestClient(app)

def test_api_startup_and_endpoints():
    """
    Simulates startup events, triggers model training, and tests main API endpoints.
    """
    # Test FastAPI Client context manager triggers the @app.on_event("startup")
    with TestClient(app) as tc:
        # 1. Test /recommend Endpoint
        req_payload = {
            "user_id": 42,
            "top_n": 5
        }
        res = tc.post("/recommend", json=req_payload)
        assert res.status_code == 200
        
        data = res.json()
        assert "user_id" in data
        assert "group" in data
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 5
        
        # 2. Test /interact Ingestion Endpoint
        event_payload = {
            "user_id": 42,
            "video_id": 10,
            "click": 1,
            "watch_ratio": 0.85,
            "like": 1
        }
        res_interact = tc.post("/interact", json=event_payload)
        assert res_interact.status_code == 200
        assert res_interact.json()["status"] == "success"
        
        # 3. Test /metrics Dashboard Endpoint
        res_metrics = tc.get("/metrics")
        assert res_metrics.status_code == 200
        metrics_data = res_metrics.json()
        assert "experiment_name" in metrics_data
        assert "metrics" in metrics_data
        assert "Control" in metrics_data["metrics"]
        assert "Treatment" in metrics_data["metrics"]
        
        # Check if click was logged in metrics
        cohort = data["group"]
        assert metrics_data["metrics"][cohort]["clicks"] >= 1
