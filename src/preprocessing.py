"""
Scikit-Learn ColumnTransformer and Pipeline Preprocessing Builder.
Ensures zero data leakage by encapsulating fit-transform logic exclusively inside cross-validation folds.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml

def get_preprocessing_pipeline(config=None):
    """
    Constructs a ColumnTransformer for numeric and categorical feature processing.
    - Numeric: Median Imputation + StandardScaler
    - Categorical: Most-Frequent Imputation + OneHotEncoder (handle_unknown='ignore')
    """
    if config is None:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
    numeric_features = list(config['features']['numeric'].keys())
    categorical_features = list(config['features']['categorical'].keys())
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features
