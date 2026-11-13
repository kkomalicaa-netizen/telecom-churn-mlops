from fastapi.testclient import TestClient
from app import app, API_KEY

client = TestClient(app)
headers = {"x-api-key": API_KEY}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info_unauthorized():
    response = client.get("/model-info") # No API key
    assert response.status_code == 401

def test_model_info_authorized():
    response = client.get("/model-info", headers=headers)
    # Will be 200 if model metadata exists, or handled gracefully
    assert response.status_code in [200, 404]

def test_predict_validation_error():
    # Sending invalid data types to test Pydantic rejection (Requirement 5)
    invalid_payload = {"Tenure Months": "not-an-integer"}
    response = client.post("/predict", json=invalid_payload, headers=headers)
    assert response.status_code == 422 # Unprocessable Entity

def test_drift_check():
    payload = {"recent_tenure_months": [2, 5, 12, 24, 36]}
    response = client.post("/drift-check", json=payload, headers=headers)
    assert response.status_code == 200
    assert "population_stability_index" in response.json()