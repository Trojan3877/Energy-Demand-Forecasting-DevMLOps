from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"sequence": [[1,2,3]]})
    assert response.status_code == 200
