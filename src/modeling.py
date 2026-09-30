"""
Machine Learning Model Definitions, Candidate Comparison, and Tuning Module.
Includes foundational classifiers (Dummy, Logistic Regression, KNN, Decision Tree) and RandomForest comparison.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
from src.preprocessing import get_preprocessing_pipeline

def build_candidate_pipelines(config=None):
    """
    Returns a dictionary of (unfitted) full model pipelines: Preprocessor + Classifier
    """
    preprocessor, _, _ = get_preprocessing_pipeline(config)
    
    candidates = {
        'Dummy': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DummyClassifier(strategy='most_frequent'))
        ]),
        'LogisticRegression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'KNN': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'DecisionTree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'))
        ]),
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced'))
        ])
    }
    
    return candidates

def get_hyperparameter_param_grids():
    """
    Defines hyperparameter tuning grids for GridSearchCV.
    """
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs']
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        },
        'DecisionTree': {
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [4, 6, 8]
        }
    }
    return param_grids

def train_and_evaluate_candidates(X_train, y_train, config=None):
    """
    Performs 5-Fold Stratified Cross-Validation for all candidate models.
    Executes GridSearchCV hyperparameter tuning where appropriate.
    Returns:
      - cv_results_df: Pandas DataFrame summarizing mean & std CV metrics for each candidate model.
      - best_candidate_name: Name of the winning candidate based on Macro F1 score.
      - fitted_best_pipeline: Winning fitted Pipeline refitted on full training data.
    """
    candidates = build_candidate_pipelines(config)
    param_grids = get_hyperparameter_param_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_score = -1.0
    best_name = None
    fitted_best_pipeline = None
    
    for name, pipeline in candidates.items():
        if name in param_grids:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            mean_f1 = grid_search.best_score_
            std_f1 = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            best_params = str(grid_search.best_params_)
        else:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid={},
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            mean_f1 = grid_search.best_score_
            std_f1 = grid_search.cv_results_['std_test_score'][0]
            best_params = "Default"
            
        results.append({
            'Model': name,
            'CV_Macro_F1_Mean': round(float(mean_f1), 4),
            'CV_Macro_F1_Std': round(float(std_f1), 4),
            'Best_Parameters': best_params
        })
        
        if mean_f1 > best_score:
            best_score = mean_f1
            best_name = name
            fitted_best_pipeline = best_model
            
    # Refit the overall best pipeline on full training dataset
    fitted_best_pipeline.fit(X_train, y_train)
    cv_df = pd.DataFrame(results).sort_values(by='CV_Macro_F1_Mean', ascending=False).reset_index(drop=True)
    
    return cv_df, best_name, fitted_best_pipeline
