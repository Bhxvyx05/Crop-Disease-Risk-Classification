"""
Model Evaluation CLI Script for Crop Disease Risk Classification.
Evaluates the saved pipeline ONCE on the held-out test set, saves metrics reports,
and generates confusion matrix & feature importance visualizations.
"""

import os
import argparse
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics import compute_evaluation_metrics
from src.explainability import get_global_feature_importance, get_feature_names_from_pipeline

def main(test_data_path="data/processed/test.csv", pipeline_path="artifacts/model_pipeline.joblib", metadata_path="artifacts/model_metadata.json"):
    print("=" * 60)
    print(" CROP DISEASE RISK CLASSIFICATION - MODEL EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load test dataset
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found at '{test_data_path}'. Please run model training first.")
    df_test = pd.read_csv(test_data_path)
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    target_col = config['dataset']['target_col']
    target_classes = config['dataset']['target_classes']
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    # Load pipeline
    print(f"\n[Step 1/4] Loading serialized model pipeline from '{pipeline_path}'...")
    pipeline = joblib.load(pipeline_path)
    
    # Predict on test set
    print("\n[Step 2/4] Generating predictions on held-out test set...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    print("\n[Step 3/4] Calculating evaluation metrics...")
    test_metrics = compute_evaluation_metrics(y_test, y_pred, target_classes=target_classes)
    
    print("\n--- HELD-OUT TEST PERFORMANCE METRICS ---")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision:   {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:      {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    
    print("\n--- Per-Class Breakdown ---")
    for cls_name, cls_m in test_metrics['per_class'].items():
        print(f" Class '{cls_name:<8}': Precision={cls_m['precision']:.4f}, Recall={cls_m['recall']:.4f}, F1={cls_m['f1_score']:.4f}, Support={cls_m['support']}")
        
    # Save metrics JSON reports
    os.makedirs("reports", exist_ok=True)
    with open(config['paths']['metrics_report'], 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved metrics report at '{config['paths']['metrics_report']}'.")
    
    # Update model metadata with test results
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['test_metrics'] = test_metrics
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata artifact at '{metadata_path}'.")
        
    # Step 4: Plot & Save Confusion Matrix & Feature Importance
    print("\n[Step 4/4] Generating evaluation plot artifacts...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm_array = np.array(test_metrics['confusion_matrix'])
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_classes, yticklabels=target_classes, cbar=False)
    plt.title('Test Set Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Label', fontweight='bold', labelpad=8)
    plt.ylabel('Actual True Label', fontweight='bold', labelpad=8)
    plt.tight_layout()
    cm_fig_path = "reports/figures/confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure at '{cm_fig_path}'.")
    
    # 2. Feature Importance Plot
    imp_df, metric_name = get_global_feature_importance(pipeline)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df.head(8), x='Importance', y='Feature', color='#2E7D32')
    plt.title(f'Global Feature Importance ({metric_name})', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel(metric_name, fontweight='bold')
    plt.tight_layout()
    imp_fig_path = "reports/figures/feature_importance.png"
    plt.savefig(imp_fig_path, dpi=300)
    plt.close()
    print(f"Saved feature importance figure at '{imp_fig_path}'.")
    
    print("\nModel evaluation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Crop Disease Risk Model")
    parser.add_argument("--test-data", default="data/processed/test.csv", help="Path to processed test CSV")
    parser.add_argument("--pipeline", default="artifacts/model_pipeline.joblib", help="Path to saved pipeline")
    parser.add_argument("--metadata", default="artifacts/model_metadata.json", help="Path to model metadata")
    args = parser.parse_args()
    
    main(args.test_data, args.pipeline, args.metadata)
