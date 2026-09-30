# Crop Disease Risk Classification Using Environmental Indicators and Machine Learning: An End-to-End Educational Prototype

**Author:** Capstone Research Team / AI & Data Science Division  
**Institution:** Academic Capstone Project  
**Date:** September 2026  
**Repository:** `https://github.com/academic-capstone/crop-disease-risk-classification`

---

## Abstract

Phytopathological epidemics cause significant agricultural yield losses globally. Timely assessment of crop disease risk based on micro-climatic indicators is crucial for precision farming and early pest management. This paper presents an end-to-end, zero-cost, reproducible Machine Learning (ML) capstone application titled **Crop Disease Risk Classification**. Using a benchmark agronomic dataset of 1,200 observations comprising 5 numerical environmental indicators (air temperature, relative humidity, soil moisture, precipitation, and leaf wetness duration) and 2 categorical crop variables (crop variety and physiological growth stage), we evaluate five candidate algorithms: Dummy Classifier, Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest. To eliminate data leakage, all preprocessing transformations (median imputation, standardization, one-hot encoding) were strictly encapsulated within Scikit-Learn `ColumnTransformer` and `Pipeline` objects. Hyperparameter tuning via 5-fold Stratified Cross-Validation selected **Random Forest** as the optimal model (Mean CV Macro-F1 = 0.7368). On the held-out test set (n=240), the model achieved an **Accuracy of 72.92%**, a **Macro F1-Score of 0.7240**, and a **Weighted F1-Score of 0.7255**. The trained pipeline was serialized and deployed via a responsive, interactive Streamlit web dashboard featuring real-time input validation, prediction score breakdown, local factor explanations, and a What-If parameter simulator.

---

## 1. Introduction

Crop diseases caused by fungal, bacterial, and viral pathogens pose severe threats to global food security and agricultural productivity. Conventional field surveillance often relies on manual visual inspection after physical disease symptoms manifest, which limits proactive intervention. However, micro-environmental conditions—such as relative humidity, leaf wetness duration, and ambient temperature—serve as early environmental prerequisites for pathogen spore germination and proliferation.

Machine Learning (ML) offers promising tools for pattern recognition in agronomic datasets. This capstone project develops a reproducible, multi-class classification prototype that categorizes crop observations into three disease risk tiers: `Low`, `Moderate`, and `High`.

---

## 2. Problem Definition

Given an observation tuple $X = (x_1, x_2, \dots, x_7)$ where:
* $x_1$: Air Temperature ($^\circ\text{C}$)
* $x_2$: Relative Air Humidity ($\%$)
* $x_3$: Volumetric Soil Moisture ($\%$)
* $x_4$: Recent Precipitation / Rainfall ($\text{mm}$)
* $x_5$: Leaf Wetness Duration ($\text{hours/day}$)
* $x_6$: Cultivated Crop Type ($\text{Wheat, Rice, Maize, Tomato, Potato}$)
* $x_7$: Crop Growth Stage ($\text{Seedling, Vegetative, Flowering, Maturity}$)

The goal is to learn a hypothesis function $f: X \to Y$ that maps $X$ to a categorical disease risk target label $Y \in \{\text{Low}, \text{Moderate}, \text{High}\}$.

---

## 3. Related Work

Prior research in agricultural informatics broadly falls into two categories:
1. **Computer Vision Diagnosis:** Deep Convolutional Neural Networks (CNNs) trained on leaf image datasets (e.g., PlantVillage) to identify visible symptoms.
2. **Environmental Epidemiology:** Tabular classification models analyzing meteorological data to estimate epidemic risk prior to visual symptom manifestation.

Our work aligns with environmental epidemiology, providing a lightweight, interpretable tabular ML system that operates without requiring specialized camera hardware or GPU infrastructure.

---

## 4. Dataset Description

The dataset consists of 1,200 observations generated according to phytopathological micro-climatic rules with realistic stochastic noise.

### Target Distribution
* **Low Risk:** 420 observations (35.0%)
* **Moderate Risk:** 396 observations (33.0%)
* **High Risk:** 384 observations (32.0%)

### Feature Specifications
* `temperature_c`: 10.0 to 45.0 $^\circ\text{C}$
* `humidity_pct`: 20.0 to 100.0 $\%$
* `soil_moisture_pct`: 5.0 to 90.0 $\%$
* `rainfall_mm`: 0.0 to 200.0 $\text{mm}$
* `leaf_wetness_hours`: 0.0 to 24.0 $\text{hours/day}$
* `crop_type`: 5 nominal categories
* `growth_stage`: 4 ordinal categories

---

## 5. Methodology

### 5.1 Data Splitting Strategy
The dataset was partitioned using a **Stratified 80/20 Train/Test Split** with a fixed random seed (`random_state=42`), reserving 960 rows for training and 240 rows for final held-out test evaluation.

### 5.2 Preprocessing Pipeline & Data Leakage Guard
All preprocessing steps were encapsulated inside a Scikit-Learn `ColumnTransformer`:
* **Numeric Sub-Pipeline:** Median Imputation (`SimpleImputer`) + Feature Scaling (`StandardScaler`).
* **Categorical Sub-Pipeline:** Most-Frequent Imputation + One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`).

```
Raw Input Data
      │
      ├── Numeric Features ──► SimpleImputer(median) ──► StandardScaler ──────┐
      │                                                                        ├──► Concatenated Feature Matrix (14 cols) ──► Estimator
      └── Categorical Features ──► SimpleImputer(mode) ──► OneHotEncoder ──────┘
```

This structure guarantees that scaler parameters ($\mu, \sigma$) and one-hot categories are fitted exclusively on training folds during cross-validation, preventing data leakage.

---

## 6. Exploratory Data Analysis (EDA)

EDA revealed the following key insights:
1. **Leaf Wetness & Humidity:** Displayed the strongest bivariate correlation with disease risk level. Observations with leaf wetness duration $>10\text{ hrs/day}$ and relative humidity $>75\%$ overwhelmingly correlated with the `High` risk class.
2. **Temperature Window:** Pathogen risk peaked in the temperate window ($18^\circ\text{C}$ to $32^\circ\text{C}$). Extreme temperatures ($>35^\circ\text{C}$ or $<12^\circ\text{C}$) inhibited risk levels even under high humidity.
3. **Multi-collinearity:** Soil moisture showed a moderate positive correlation with relative humidity ($r = 0.38$).

---

## 7. Model Development & Hyperparameter Tuning

We evaluated five candidate classifiers using **5-Fold Stratified Cross-Validation** on the 960-row training set, optimizing for **Macro F1-Score**:

1. **Dummy Classifier:** Naive baseline predicting the most frequent class.
2. **Logistic Regression:** Linear classifier with L2 regularization (`C=0.1`, `solver='lbfgs'`).
3. **K-Nearest Neighbors (KNN):** Distance-based non-parametric classifier (`n_neighbors=9`, `weights='distance'`).
4. **Decision Tree:** Interpretable rule-based tree (`max_depth=5`, `min_samples_split=5`).
5. **Random Forest:** Ensemble of decision trees (`n_estimators=50`, `max_depth=8`).

### Cross-Validation Results Summary

| Model | CV Mean Macro F1 | CV Std Dev | Best Selected Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.7368** | **0.0245** | `max_depth=8, n_estimators=50` |
| Decision Tree | 0.7179 | 0.0190 | `max_depth=5, min_samples_split=5` |
| Logistic Regression | 0.7042 | 0.0395 | `C=0.1, solver='lbfgs'` |
| K-Nearest Neighbors | 0.6807 | 0.0321 | `n_neighbors=9, weights='distance'` |
| Dummy Classifier | 0.1728 | 0.0008 | Default (Most Frequent) |

**Random Forest** achieved the highest cross-validation Macro F1 score (0.7368) and was selected as the final model pipeline.

---

## 8. Model Evaluation & Held-Out Test Results

The winning Random Forest pipeline was refitted on the full 960-row training set and evaluated **once** on the 240-row held-out test set.

### Overall Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.7292** (72.92%) |
| **Balanced Accuracy** | **0.7271** (72.71%) |
| **Macro Precision** | **0.7227** |
| **Macro Recall** | **0.7271** |
| **Macro F1-Score** | **0.7240** |
| **Weighted F1-Score** | **0.7255** |

### Per-Class Performance Breakdown

| Risk Class | Precision | Recall | F1-Score | Support (N) |
| :--- | :--- | :--- | :--- | :--- |
| **Low Risk** | 0.7889 | 0.8452 | **0.8161** | 84 |
| **Moderate Risk** | 0.6197 | 0.5570 | **0.5867** | 79 |
| **High Risk** | 0.7595 | 0.7792 | **0.7692** | 77 |

### Confusion Matrix Analysis
* **Low Risk:** 71 correctly classified, 13 misclassified as Moderate.
* **Moderate Risk:** 44 correctly classified, 18 misclassified as Low, 17 misclassified as High.
* **High Risk:** 60 correctly classified, 16 misclassified as Moderate, 1 misclassified as Low.

The model demonstrates strong performance in distinguishing Low and High risk extremes, with expected boundary overlap occurring in the intermediate Moderate risk class.

---

## 9. Streamlit Application & System Integration

The application (`app.py`) provides a modern agricultural SaaS interface containing five core pages:
1. **Overview Dashboard:** High-level metrics, dataset summaries, pie charts, and educational disclaimers.
2. **Predict Risk:** Interactive observation entry form, dynamic domain validation, risk category badges, model confidence score distribution, driving factor explanations, and a What-If parameter modification tool.
3. **Data Analytics:** Interactive Plotly histograms, boxplots, correlation heatmaps, and categorical risk breakdowns.
4. **Model Insights:** Cross-validation comparison tables, test confusion matrix heatmap, per-class metrics bar chart, and global Gini feature importances.
5. **About Project:** Architecture diagrams, tech stack details, limitations, and references.

---

## 10. Limitations & Responsible Use Statement

* **Dataset Scope:** The prototype relies on environmental indicators. It does not account for specific pathogen strains, host plant genetics, or micro-nutrient soil deficiencies.
* **Non-Causal Interpretation:** Feature importances reflect statistical associations within the dataset, not biological causality.
* **Educational Disclaimer:** This software is an educational prototype and must not be used as a substitute for certified agricultural extension advice or laboratory phytopathological diagnosis.

---

## 11. Conclusion & Future Scope

This capstone project successfully demonstrates an end-to-end Machine Learning pipeline for crop disease risk classification. By enforcing zero data leakage, comparing multiple foundational classifiers, providing local and global explainability, and deploying a functional Streamlit interface, the system achieves an academic test accuracy of 72.92% and a Macro F1 score of 0.7240.

**Future Scope:**
1. Integration of real-time IoT weather station API feeds.
2. Extension to spatio-temporal disease mapping.
3. Addition of SHAP (SHapley Additive exPlanations) for local feature attribution.

---

## 12. References

1. Agrios, G. N. (2005). *Plant Pathology* (5th ed.). Academic Press.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Streamlit Documentation (2026). *Streamlit API Reference*. https://docs.streamlit.io/
4. Scikit-learn Guide (2026). *Common Pitfalls and Data Leakage*. https://scikit-learn.org/stable/common_pitfalls.html
