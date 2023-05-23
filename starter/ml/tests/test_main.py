from fastapi.testclient import TestClient
#import fastapi.HTTPException as HTTPException
import json
import logging

from main import app

test_client = TestClient(app)


def test_home():
    """
    Test welcome message for GET at root
    """
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello world from model API..."


def test_model_inference():
    """
    Test model inference output
    """
    sample_data = {
        'age': 31,
        'workclass': "Private",
        'fnlgt': 45781,
        'education': "Masters",
        'education_num': 14,
        'marital_status': "Never-married",
        'occupation': "Prof-specialty",
        'relationship': "Not-in-family",
        'race': "White",
        'sex': "Female",
        'capital_gain': 14084,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': "United-States"
    }

    data = json.dumps(sample_data)

    response = test_client.post("/inference/", data=data)

    # Test response and output
    assert response.status_code == 200
    assert response.json()["age"] == 50
    assert response.json()["fnlgt"] == 234721

    # Test prediction vs expected label
    logging.info(f'+++++++ prediction = {response.json()["prediction"]} +++++++')
    assert response.json()["prediction"] == '>50K'


def test_model_inference_class0():
    """
    Test model inference output for class 0
    """
    sample_data = {
        'age': 39,
        'workclass': "State-gov",
        'fnlgt': 77516,
        'education': "Bachelors",
        'education_num': 13,
        'marital_status': "Never-married",
        'occupation': "Adm-clerical",
        'relationship': "Not-in-family",
        'race': "White",
        'sex': "Male",
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': "United-States"
    }

    data = json.dumps(sample_data)

    response = test_client.post("/inference/", data=data)

    # Test response and output
    assert response.status_code == 200
    assert response.json()["age"] == 30
    assert response.json()["fnlgt"] == 234721

    # Test prediction vs expected label
    logging.info(f'+++++++ prediction = {response.json()["prediction"]} +++++++')
    assert response.json()["prediction"][0] == '<=50K'


def test_wrong_inference_query():
    """
    Test incomplete sample that does not generate a prediction
    """
    sample_data = {
        'age': 50,
        'workclass': "Private",
        'fnlgt': 234721
    }

    data = json.dumps(sample_data)
    response = test_client.post("/inference/", data=data)

    assert 'prediction' not in response.json().keys()
    logging.warning(f"The sample has {len(sample_data)} features. It must have 14 features.")


if __name__ == '__main__':
    test_home()
    test_model_inference()
    test_model_inference_class0()
    test_wrong_inference_query()