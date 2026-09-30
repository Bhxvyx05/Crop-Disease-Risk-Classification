"""
Data Schema, Input Validation, and Quality Checks Module for Crop Disease Risk Classification.
"""

import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_raw_dataset(df, config=None):
    """
    Audits raw dataframe against expected schema, feature types, target classes,
    missing values, duplicate rows, and invalid ranges.
    Returns a dictionary summarizing validation results and clean status.
    """
    if config is None:
        config = load_config()
        
    target_col = config['dataset']['target_col']
    expected_classes = set(config['dataset']['target_classes'])
    numeric_features = list(config['features']['numeric'].keys())
    categorical_features = list(config['features']['categorical'].keys())
    expected_cols = numeric_features + categorical_features + [target_col]
    
    issues = []
    
    # 1. Column existence check
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")
        
    # 2. Target validity check
    if target_col in df.columns:
        actual_classes = set(df[target_col].dropna().unique())
        unexpected_classes = actual_classes - expected_classes
        if unexpected_classes:
            issues.append(f"Unexpected target classes found: {unexpected_classes}")
    else:
        issues.append(f"Target column '{target_col}' not found.")
        
    # 3. Data type check
    for col in numeric_features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column '{col}' expected numeric, got {df[col].dtype}")
            
    # 4. Duplicate rows check
    num_duplicates = int(df.duplicated().sum())
    
    # 5. Missing values audit
    missing_counts = df[expected_cols].isnull().sum().to_dict() if all(c in df.columns for c in expected_cols) else {}
    
    validation_report = {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'num_records': len(df),
        'num_features': len(numeric_features) + len(categorical_features),
        'duplicates_count': num_duplicates,
        'missing_counts': missing_counts,
        'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else {}
    }
    
    return validation_report

def validate_single_observation(input_dict, config=None):
    """
    Validates a single input observation dictionary for real-time inference.
    Returns (is_valid, error_messages_list).
    """
    if config is None:
        config = load_config()
        
    errors = []
    num_cfg = config['features']['numeric']
    cat_cfg = config['features']['categorical']
    
    # Validate numerical inputs
    for feat, meta in num_cfg.items():
        if feat not in input_dict or input_dict[feat] is None or pd.isna(input_dict[feat]):
            errors.append(f"Field '{feat}' is required.")
            continue
        try:
            val = float(input_dict[feat])
            min_v, max_v = meta['min'], meta['max']
            if val < min_v or val > max_v:
                errors.append(f"'{meta['description']}' ({feat}) value {val} is outside valid domain range [{min_v}, {max_v}] {meta['unit']}.")
        except (ValueError, TypeError):
            errors.append(f"Field '{feat}' must be a valid number.")
            
    # Validate categorical inputs
    for feat, meta in cat_cfg.items():
        if feat not in input_dict or not input_dict[feat]:
            errors.append(f"Field '{feat}' is required.")
            continue
        val = str(input_dict[feat])
        valid_cats = meta['categories']
        if val not in valid_cats:
            errors.append(f"Invalid selection '{val}' for '{feat}'. Supported categories: {valid_cats}.")
            
    return len(errors) == 0, errors

if __name__ == "__main__":
    df = pd.read_csv("data/raw/crop_disease_risk.csv")
    report = validate_raw_dataset(df)
    print("Dataset Validation Report:")
    print(report)
