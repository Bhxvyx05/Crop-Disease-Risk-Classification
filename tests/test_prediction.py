"""
Integration tests for real-time model inference and prediction pipeline.
"""

from src.inference import ModelInferencePipeline

def test_inference_pipeline_prediction():
    pipeline = ModelInferencePipeline()
    sample_obs = {
        'temperature_c': 24.5,
        'humidity_pct': 85.0,
        'soil_moisture_pct': 60.0,
        'rainfall_mm': 55.0,
        'leaf_wetness_hours': 14.0,
        'crop_type': 'Tomato',
        'growth_stage': 'Flowering'
    }
    
    result = pipeline.predict_single(sample_obs)
    
    assert result['success'] is True
    assert result['prediction'] in ['Low', 'Moderate', 'High']
    assert result['class_probabilities'] is not None
    assert set(result['class_probabilities'].keys()) == {'Low', 'Moderate', 'High'}
    assert len(result['explanation']['agronomic_factors']) > 0
