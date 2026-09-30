"""
Model Explainability and Local / Global Prediction Interpretation Module.
Provides feature importances, decision path rules, feature contribution explanations, and What-If simulation.
"""

import numpy as np
import pandas as pd

def get_feature_names_from_pipeline(fitted_pipeline):
    """
    Extracts post-preprocessing feature names from ColumnTransformer inside a fitted Pipeline.
    """
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    
    cat_onehot = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_encoded_names = cat_onehot.get_feature_names_out(cat_features).tolist()
    
    all_feature_names = list(num_features) + cat_encoded_names
    return all_feature_names

def get_global_feature_importance(fitted_pipeline):
    """
    Extracts global feature importances (or coefficient magnitudes) from the fitted classifier.
    Returns a sorted DataFrame of feature importances.
    """
    feature_names = get_feature_names_from_pipeline(fitted_pipeline)
    classifier = fitted_pipeline.named_steps['classifier']
    
    importances = None
    metric_name = "Importance"
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        metric_name = "Gini Importance"
    elif hasattr(classifier, 'coef_'):
        # For multi-class Logistic Regression, average absolute coefficient across classes
        importances = np.mean(np.abs(classifier.coef_), axis=0)
        metric_name = "Mean Abs Coefficient"
    else:
        # Fallback uniform importances for KNN/Dummy
        importances = np.ones(len(feature_names)) / len(feature_names)
        metric_name = "Uniform Proxy"
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    df_imp['Normalized_Importance'] = df_imp['Importance'] / df_imp['Importance'].sum()
    return df_imp, metric_name

def explain_single_prediction(fitted_pipeline, input_df):
    """
    Generates a human-readable explanation of key factors driving a single prediction observation.
    """
    feature_imp_df, _ = get_global_feature_importance(fitted_pipeline)
    top_features = feature_imp_df.head(3)['Feature'].tolist()
    
    obs = input_df.iloc[0]
    
    explanation_points = []
    
    # Check key agronomic thresholds for qualitative interpretation
    if 'leaf_wetness_hours' in obs and obs['leaf_wetness_hours'] > 8.0:
        explanation_points.append(f"Leaf wetness duration is elevated ({obs['leaf_wetness_hours']} hrs/day), which heavily promotes pathogen infection.")
    if 'humidity_pct' in obs and obs['humidity_pct'] > 70.0:
        explanation_points.append(f"Relative humidity is high ({obs['humidity_pct']}%), creating favorable micro-climatic humidity for fungal spores.")
    if 'temperature_c' in obs:
        if 18.0 <= obs['temperature_c'] <= 32.0:
            explanation_points.append(f"Air temperature ({obs['temperature_c']}°C) lies in the optimal temperature proliferation window for leaf diseases.")
        elif obs['temperature_c'] > 32.0 or obs['temperature_c'] < 15.0:
            explanation_points.append(f"Air temperature ({obs['temperature_c']}°C) is outside peak disease proliferation bounds.")
    if 'growth_stage' in obs and obs['growth_stage'] in ["Flowering", "Vegetative"]:
        explanation_points.append(f"Crop is in the sensitive '{obs['growth_stage']}' growth stage, increasing biological vulnerability.")
        
    if not explanation_points:
        explanation_points.append("Environmental metrics indicate low humidity and short leaf wetness, mitigating disease pathogen pressure.")
        
    return {
        'top_model_features': top_features,
        'agronomic_factors': explanation_points
    }

def simulate_what_if(fitted_pipeline, base_input_df, modified_feature, new_value):
    """
    Simulates a 'What-If' scenario by modifying a single input variable and returning the original vs modified risk prediction.
    """
    modified_df = base_input_df.copy()
    modified_df[modified_feature] = new_value
    
    orig_pred = fitted_pipeline.predict(base_input_df)[0]
    mod_pred = fitted_pipeline.predict(modified_df)[0]
    
    orig_proba = None
    mod_proba = None
    if hasattr(fitted_pipeline, 'predict_proba'):
        classes = fitted_pipeline.classes_.tolist()
        orig_p = fitted_pipeline.predict_proba(base_input_df)[0]
        mod_p = fitted_pipeline.predict_proba(modified_df)[0]
        orig_proba = dict(zip(classes, [round(float(p), 4) for p in orig_p]))
        mod_proba = dict(zip(classes, [round(float(p), 4) for p in mod_p]))
        
    return {
        'modified_feature': modified_feature,
        'original_value': base_input_df.iloc[0][modified_feature],
        'new_value': new_value,
        'original_prediction': orig_pred,
        'modified_prediction': mod_pred,
        'original_probabilities': orig_proba,
        'modified_probabilities': mod_proba
    }
