"""
Model Inference and Real-Time Prediction Pipeline Module.
Loads serialized fitted pipeline, validates single observation inputs, and returns risk prediction & explanations.
"""

import os
import joblib
import json
import pandas as pd
from src.data_validation import validate_single_observation, load_config
from src.explainability import explain_single_prediction

class ModelInferencePipeline:
    def __init__(self, pipeline_path="artifacts/model_pipeline.joblib", metadata_path="artifacts/model_metadata.json"):
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = None
        self.config = load_config()
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Model pipeline artifact not found at '{self.pipeline_path}'. Please run model training first.")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Model metadata artifact not found at '{self.metadata_path}'. Please run model training first.")
            
        self.pipeline = joblib.load(self.pipeline_path)
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
    def predict_single(self, input_dict):
        """
        Validates input observation dictionary and returns prediction result, class probabilities, and explanation.
        """
        is_valid, errors = validate_single_observation(input_dict, self.config)
        if not is_valid:
            return {
                'success': False,
                'errors': errors,
                'prediction': None
            }
            
        # Convert input dictionary into DataFrame matching exact training feature order
        numeric_features = list(self.config['features']['numeric'].keys())
        categorical_features = list(self.config['features']['categorical'].keys())
        feature_order = numeric_features + categorical_features
        
        input_df = pd.DataFrame([input_dict])[feature_order]
        
        # Predict class
        predicted_class = str(self.pipeline.predict(input_df)[0])
        
        # Predict probabilities if supported
        class_probabilities = None
        if hasattr(self.pipeline, 'predict_proba'):
            probs = self.pipeline.predict_proba(input_df)[0]
            classes = self.pipeline.classes_.tolist()
            class_probabilities = {
                cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)
            }
            
        # Generate explanation
        explanation = explain_single_prediction(self.pipeline, input_df)
        
        return {
            'success': True,
            'prediction': predicted_class,
            'class_probabilities': class_probabilities,
            'explanation': explanation,
            'model_name': self.metadata.get('model_name', 'Fitted Model'),
            'test_accuracy': self.metadata.get('test_metrics', {}).get('accuracy', 'N/A'),
            'test_macro_f1': self.metadata.get('test_metrics', {}).get('macro_f1', 'N/A')
        }
