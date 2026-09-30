"""
Unit tests for data schema and input validation functions.
"""

import pandas as pd
from src.data_validation import validate_raw_dataset, validate_single_observation

def test_validate_raw_dataset_success():
    df = pd.read_csv("data/raw/crop_disease_risk.csv")
    report = validate_raw_dataset(df)
    assert report['is_valid'] is True
    assert report['num_records'] == 1200
    assert report['num_features'] == 7

def test_validate_single_observation_valid():
    valid_input = {
        'temperature_c': 25.0,
        'humidity_pct': 80.0,
        'soil_moisture_pct': 50.0,
        'rainfall_mm': 30.0,
        'leaf_wetness_hours': 10.0,
        'crop_type': 'Tomato',
        'growth_stage': 'Flowering'
    }
    is_valid, errors = validate_single_observation(valid_input)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_single_observation_invalid_range():
    invalid_input = {
        'temperature_c': 99.0, # Exceeds domain max 45.0
        'humidity_pct': 80.0,
        'soil_moisture_pct': 50.0,
        'rainfall_mm': 30.0,
        'leaf_wetness_hours': 10.0,
        'crop_type': 'Tomato',
        'growth_stage': 'Flowering'
    }
    is_valid, errors = validate_single_observation(invalid_input)
    assert is_valid is False
    assert len(errors) > 0
    assert any("outside valid domain range" in err for err in errors)
