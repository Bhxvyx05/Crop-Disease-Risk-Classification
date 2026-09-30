# Capstone Presentation Outline & Viva Defense Guide

## Part 1: Presentation Outline (10 Slides)

### Slide 1: Title & Overview
* **Title:** Crop Disease Risk Classification
* **Subtitle:** An End-to-End Supervised ML Pipeline with Explainable Web Interface
* **Presenter:** Student / Capstone Researcher
* **Key Focus:** Environmental indicator analysis for phytopathology risk assessment.

### Slide 2: Problem Statement & Motivation
* **Problem:** Crop disease outbreaks cause severe agricultural loss; visual diagnosis is reactive.
* **Objective:** Build an ML application that classifies micro-climatic risk into `Low`, `Moderate`, and `High` tiers.
* **Scope:** Open-source Python ML stack, zero data leakage, functional Streamlit prototype.

### Slide 3: Dataset & Feature Schema
* **Total Dataset Size:** 1,200 observations (80/20 train/test split).
* **Numerical Features (5):** Temperature (°C), Relative Humidity (%), Soil Moisture (%), Rainfall (mm), Leaf Wetness (hrs/day).
* **Categorical Features (2):** Crop Type (Wheat, Rice, Maize, Tomato, Potato), Growth Stage (Seedling, Vegetative, Flowering, Maturity).
* **Target Variable:** `disease_risk` (`Low`: 35%, `Moderate`: 33%, `High`: 32%).

### Slide 4: Data Quality & Preprocessing
* **Missing Values:** Median imputation for numerical metrics; mode imputation for categorical metrics.
* **Scaling & Encoding:** `StandardScaler` for numerical metrics; `OneHotEncoder` for nominal categories.
* **Leakage Guard:** Encapsulated inside `ColumnTransformer` and `Pipeline`, fitted strictly on training folds during cross-validation.

### Slide 5: Methodology & Model Candidates
* **5-Fold Stratified Cross-Validation:** Preserves class balance across all folds.
* **Models Evaluated:** Dummy Classifier (baseline), Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest.
* **Optimization Metric:** Macro F1-Score (gives equal weight to all three risk tiers).

### Slide 6: Model Selection & Comparison
* **CV Macro F1 Comparison:**
  1. **Random Forest:** 0.7368 (Selected Winner)
  2. Decision Tree: 0.7179
  3. Logistic Regression: 0.7042
  4. K-Nearest Neighbors: 0.6807
  5. Dummy Classifier: 0.1728
* **Hyperparameters Selected:** `n_estimators=50, max_depth=8`.

### Slide 7: Held-Out Test Evaluation Results
* **Test Accuracy:** 72.92%
* **Balanced Accuracy:** 72.71%
* **Macro F1-Score:** 0.7240
* **Per-Class F1:** Low (0.8161), High (0.7692), Moderate (0.5867).

### Slide 8: Streamlit Application Demo
* **5 Core Pages:** Overview Dashboard, Predict Risk, Data Analytics, Model Insights, About Project.
* **Key Features:** Form validation, risk badges, estimated class scores, local factor explanations, What-If simulator.

### Slide 9: Limitations & Responsible Use
* **Dataset Boundary:** Simulated environmental micro-climates; lacks specific laboratory strain genomics.
* **Non-Causal Note:** Explanations represent statistical reliance, not biological proof.
* **Disclaimer:** Educational prototype; not a certified diagnostic or pesticide recommendation tool.

### Slide 10: Conclusion & Future Scope
* **Key Takeaway:** Successfully built a reproducible, end-to-end ML project meeting all 10-day capstone criteria.
* **Future Scope:** Real-time weather station API integration and SHAP local feature attribution.

---

## Part 2: Top 15 Viva Defense Questions & Answers

### Q1: What is the exact Machine Learning task solved in this project?
**Answer:** It is a multi-class supervised classification task. The input vector $X$ consists of 7 environmental and crop indicators, and the output $y$ is a discrete target class label $\in \{\text{Low}, \text{Moderate}, \text{High}\}$.

### Q2: Why did you perform Exploratory Data Analysis (EDA) before model training?
**Answer:** EDA allows us to inspect dataset quality, verify schema and target labels, detect missing values or duplicate rows, assess class balance, understand feature distributions, and ensure there is no data leakage before fitting models.

### Q3: What is data leakage, and how did your code prevent it?
**Answer:** Data leakage occurs when information from the validation or held-out test set improperly influences the model training or preprocessing pipeline. We prevented it by placing all imputers, scalers, and encoders inside a Scikit-Learn `ColumnTransformer` wrapped in a `Pipeline`. Fitting occurs strictly on training folds during 5-fold cross-validation.

### Q4: Why did you use Stratified Train/Test Split and Stratified K-Fold Cross-Validation?
**Answer:** Stratification ensures that the proportion of target risk categories (`Low`, `Moderate`, `High`) remains approximately equal across training folds and test splits, preventing class representation imbalance in smaller samples.

### Q5: Why did you use Macro F1-Score as your primary model selection metric instead of raw Accuracy?
**Answer:** Raw accuracy can be misleading if class distributions are imbalanced or if minority classes fail. Macro F1 computes the F1-score independently for each class (`Low`, `Moderate`, `High`) and averages them equally, ensuring the model performs well across all risk tiers.

### Q6: What models were compared, and which model won?
**Answer:** We compared a Dummy Classifier baseline, Logistic Regression, KNN, Decision Tree, and Random Forest. Random Forest won with the highest 5-Fold CV Macro F1 score of **0.7368**, and achieved a held-out test accuracy of **72.92%** and Macro F1 of **0.7240**.

### Q7: Why does Logistic Regression or KNN require feature scaling (`StandardScaler`), whereas Decision Trees do not?
**Answer:** Logistic Regression uses gradient-based optimization on coefficients, and KNN uses Euclidean distance calculations; both are sensitive to feature magnitude differences. Decision Trees use monotonic threshold splits on single features, making them invariant to scale transformations.

### Q8: What hyperparameters were tuned for the winning Random Forest model?
**Answer:** We tuned `n_estimators` (number of trees, selected=50) and `max_depth` (maximum tree depth, selected=8) using `GridSearchCV`.

### Q9: Does feature importance prove that leaf wetness causes crop disease?
**Answer:** No. Feature importance measures how much a model relies on a variable to reduce split impurity or classify samples. Correlation or model importance does not prove biological causality without controlled laboratory experiments.

### Q10: How does your application handle invalid user inputs in real time?
**Answer:** The `src/data_validation.py` module checks numerical inputs against documented domain bounds (e.g. Temperature must be between 10.0 and 45.0°C) and verifies categorical selections against valid training categories before inference.

### Q11: What is the purpose of the What-If Simulator in your app?
**Answer:** It allows users to modify a single input variable (e.g., reducing leaf wetness from 14 to 4 hours) while keeping other features constant, demonstrating how the model's risk prediction responds to parameter changes.

### Q12: What artifacts are saved after model training, and why?
**Answer:** 
1. `artifacts/model_pipeline.joblib`: The complete fitted pipeline (preprocessor + model).
2. `artifacts/model_metadata.json`: Model name, parameters, feature schema, class labels, and measured test metrics for UI rendering.

### Q13: Why did you build the frontend using Streamlit?
**Answer:** Streamlit allows rapid deployment of interactive Python data applications without complex JavaScript frontend infrastructure, enabling pure Python integration with Scikit-Learn, Pandas, and Plotly.

### Q14: What are the main limitations of this project?
**Answer:** The dataset reflects micro-climatic environmental indicators and does not include genetic pathogen strain testing or soil chemistry lab tests. It is an educational prototype rather than a certified field diagnostic system.

### Q15: Can this application recommend pesticide chemical treatments?
**Answer:** No. Recommending chemical treatments requires certified agronomic authorization and localized field validation. The app explicitly displays educational disclaimers.
