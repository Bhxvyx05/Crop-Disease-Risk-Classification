"""
Unit tests for Scikit-Learn preprocessing pipeline transformer.
"""

import pandas as pd
from src.preprocessing import get_preprocessing_pipeline

def test_preprocessing_pipeline_fit_transform():
    df = pd.read_csv("data/raw/crop_disease_risk.csv")
    X = df.drop(columns=['disease_risk'])
    
    preprocessor, num_cols, cat_cols = get_preprocessing_pipeline()
    X_trans = preprocessor.fit_transform(X)
    
    # Check transformed array shape
    assert X_trans.shape[0] == len(df)
    # 5 numeric + 5 crop_type onehot + 4 growth_stage onehot = 14 columns
    assert X_trans.shape[1] == 14
