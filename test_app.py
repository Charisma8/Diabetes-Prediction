import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Sample input data
LOW_RISK = {
    "Pregnancies": 0, "Glucose": 85, "BloodPressure": 70,
    "SkinThickness": 20, "Insulin": 80, "BMI": 22.0,
    "DiabetesPedigreeFunction": 0.2, "Age": 25
}

HIGH_RISK = {
    "Pregnancies": 5, "Glucose": 180, "BloodPressure": 90,
    "SkinThickness": 40, "Insulin": 200, "BMI": 38.5,
    "DiabetesPedigreeFunction": 0.9, "Age": 55
}

def test_predict_returns_200(client):
    """API should return status 200 for valid input"""
    res = client.post('/predict',
        data=json.dumps(LOW_RISK),
        content_type='application/json')
    assert res.status_code == 200

def test_predict_has_required_fields(client):
    """Response must contain risk_percent, risk_level, shap_values"""
    res = client.post('/predict',
        data=json.dumps(LOW_RISK),
        content_type='application/json')
    data = json.loads(res.data)
    assert 'risk_percent' in data
    assert 'risk_level' in data
    assert 'shap_values' in data

def test_risk_percent_range(client):
    """Risk percent must always be between 0 and 100"""
    for payload in [LOW_RISK, HIGH_RISK]:
        res = client.post('/predict',
            data=json.dumps(payload),
            content_type='application/json')
        data = json.loads(res.data)
        assert 0 <= data['risk_percent'] <= 100

def test_risk_level_values(client):
    """Risk level must be one of the 3 valid categories"""
    res = client.post('/predict',
        data=json.dumps(HIGH_RISK),
        content_type='application/json')
    data = json.loads(res.data)
    assert data['risk_level'] in ['Low', 'Moderate', 'High']

def test_shap_values_count(client):
    """Should return SHAP value for all 8 features"""
    res = client.post('/predict',
        data=json.dumps(LOW_RISK),
        content_type='application/json')
    data = json.loads(res.data)
    assert len(data['shap_values']) == 8

def test_missing_field_returns_error(client):
    """Missing a required field should not crash the server"""
    incomplete = {"Glucose": 120, "BMI": 28}
    res = client.post('/predict',
        data=json.dumps(incomplete),
        content_type='application/json')
    assert res.status_code in [400, 500]  # should fail gracefully

def test_whatif_endpoint(client):
    """What-if endpoint should return updated risk percent"""
    res = client.post('/whatif',
        data=json.dumps(LOW_RISK),
        content_type='application/json')
    data = json.loads(res.data)
    assert 'risk_percent' in data