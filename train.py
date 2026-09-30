"""
Model Training CLI Script for Crop Disease Risk Classification.
Executes data audit, stratified train/test split, candidate model cross-validation, hyperparameter selection,
and saves the full fitted preprocessing-plus-model pipeline artifact.
"""

import os
import argparse
import yaml
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_validation import validate_raw_dataset, load_config
from src.modeling import train_and_evaluate_candidates

def main(data_path="data/raw/crop_disease_risk.csv", config_path="config/config.yaml"):
    print("=" * 60)
    print(" CROP DISEASE RISK CLASSIFICATION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    config = load_config(config_path)
    target_col = config['dataset']['target_col']
    test_size = config['dataset']['test_size']
    random_state = config['dataset']['random_state']
    
    # 1. Load dataset
    print(f"\n[Step 1/6] Loading raw dataset from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset file not found at '{data_path}'. Please check file placement.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations with {len(df.columns)} columns.")
    
    # 2. Validate raw data
    print("\n[Step 2/6] Auditing dataset schema and target validity...")
    val_report = validate_raw_dataset(df, config)
    if not val_report['is_valid']:
        print("Dataset validation warnings/issues detected:", val_report['issues'])
    else:
        print("Dataset validation passed successfully.")
    print(f"Target Distribution: {val_report['target_distribution']}")
    
    # 3. Stratified Train / Test Split
    print(f"\n[Step 3/6] Performing Stratified Train/Test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    numeric_features = list(config['features']['numeric'].keys())
    categorical_features = list(config['features']['categorical'].keys())
    feature_cols = numeric_features + categorical_features
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    os.makedirs("data/processed", exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(config['paths']['train_data'], index=False)
    test_df.to_csv(config['paths']['test_data'], index=False)
    print(f"Saved processed train set ({len(train_df)} rows) and test set ({len(test_df)} rows).")
    
    # 4. Cross-Validation and Model Selection
    print("\n[Step 4/6] Running 5-Fold Stratified Cross-Validation for candidate models...")
    cv_df, best_name, fitted_best_pipeline = train_and_evaluate_candidates(X_train, y_train, config)
    
    print("\n--- Cross-Validation Model Comparison ---")
    print(cv_df.to_string(index=False))
    print(f"\nSelected Best Model Candidate: '{best_name}' (Macro F1: {cv_df.iloc[0]['CV_Macro_F1_Mean']})")
    
    # 5. Save Artifacts
    print("\n[Step 5/6] Serializing fitted model pipeline and metadata...")
    os.makedirs("artifacts", exist_ok=True)
    pipeline_path = config['paths']['model_pipeline']
    metadata_path = config['paths']['model_metadata']
    
    joblib.dump(fitted_best_pipeline, pipeline_path)
    print(f"Saved fitted pipeline artifact at '{pipeline_path}'.")
    
    metadata = {
        'project_name': config['project']['name'],
        'version': config['project']['version'],
        'model_name': best_name,
        'target_col': target_col,
        'target_classes': config['dataset']['target_classes'],
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'random_state': random_state,
        'cv_results': cv_df.to_dict(orient='records'),
        'best_cv_macro_f1': float(cv_df.iloc[0]['CV_Macro_F1_Mean']),
        'best_parameters': cv_df.iloc[0]['Best_Parameters']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata at '{metadata_path}'.")
    
    print("\n[Step 6/6] Model training pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Disease Risk Classification Models")
    parser.add_argument("--data", default="data/raw/crop_disease_risk.csv", help="Path to raw CSV dataset")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.data, args.config)
