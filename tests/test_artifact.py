"""
Unit tests for model artifact loading and metadata integrity.
"""

import os
import json
import joblib

def test_artifacts_exist():
    assert os.path.exists("artifacts/model_pipeline.joblib")
    assert os.path.exists("artifacts/model_metadata.json")

def test_metadata_structure():
    with open("artifacts/model_metadata.json", "r") as f:
        meta = json.load(f)
        
    assert "model_name" in meta
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.0
    assert "confusion_matrix" in meta["test_metrics"]
