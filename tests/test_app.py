import io
import pytest
from PIL import Image
from src.app.app import app, db
from src.app.database import PredictionLog

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Vision" in response.data
    assert b"Upload" in response.data or b"Scan" in response.data

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data

def test_history_page_loads(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Audit Trail" in response.data

def test_reject_disallowed_file_extension(client):
    data = {
        "image": (io.BytesIO(b"fake executable code"), "malicious_script.exe")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Unsupported File Extension" in response.data

def test_reject_corrupt_image(client):
    data = {
        "image": (io.BytesIO(b"corrupt non-image header data"), "fake_image.jpg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Corrupted" in response.data or b"could not be decoded" in response.data

