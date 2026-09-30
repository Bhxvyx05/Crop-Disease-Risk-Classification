"""
Model Evaluation and Classification Metrics Module.
Calculates Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Per-Class metrics, and Confusion Matrix.
"""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import pandas as pd
import numpy as np

def compute_evaluation_metrics(y_true, y_pred, target_classes=['Low', 'Moderate', 'High']):
    """
    Computes comprehensive held-out test evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=target_classes)
    # Weighted metrics
    p_w, r_w, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=target_classes)
    
    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_true, y_pred, labels=target_classes)
    
    per_class_metrics = {}
    for i, cls_name in enumerate(target_classes):
        per_class_metrics[cls_name] = {
            'precision': round(float(p_class[i]), 4),
            'recall': round(float(r_class[i]), 4),
            'f1_score': round(float(f1_class[i]), 4),
            'support': int(support_class[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=target_classes)
    
    metrics = {
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'macro_precision': round(float(p_macro), 4),
        'macro_recall': round(float(r_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'weighted_f1': round(float(f1_weighted), 4),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'target_classes': target_classes
    }
    
    return metrics
