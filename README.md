# Crop Disease Risk Classification

An end-to-end, reproducible Machine Learning capstone application with a modern SaaS web dashboard, explainable predictions, interactive data analytics, automated tests, and academic technical documentation.

---

## 1. Project Overview

**Crop Disease Risk Classification** is a supervised Machine Learning application designed to classify crop observations into disease-risk categories (`Low`, `Moderate`, `High`) based on environmental indicators and crop characteristics.

* **Domain:** Agriculture / Phytopathology / Environmental Epidemiology
* **Task:** Multi-Class Supervised Classification
* **Dataset Size:** 1,200 records (80% Train, 20% Held-Out Test)
* **Target Categories:** `Low`, `Moderate`, `High`
* **Selected Model:** Random Forest (`n_estimators=50`, `max_depth=8`)
* **Test Accuracy:** **72.92%** | **Macro F1-Score:** **0.7240**
* **Deployment:** Interactive Streamlit SaaS Web Dashboard

---

## 2. Technology Stack

* **Machine Learning & Data Science:** Python 3.11, Scikit-Learn, Pandas, NumPy, Joblib
* **Data Visualization:** Plotly Express, Matplotlib, Seaborn
* **Frontend & Web Application:** Streamlit, Custom CSS SaaS System
* **Testing & Quality Assurance:** Pytest
* **Configuration:** PyYAML, JSON

---

## 3. Repository Directory Structure

```text
Crop Disease Risk/
├── app.py                         # Streamlit multi-page application
├── train.py                       # ML model training CLI script
├── evaluate.py                    # Test evaluation CLI script
├── requirements.txt               # Dependencies
├── README.md                      # Academic project documentation
├── DATA_CARD.md                   # Data card & feature dictionary
├── .gitignore                     # Git ignore rules
│
├── config/
│   └── config.yaml                # Project configuration file
│
├── data/
│   ├── raw/
│   │   └── crop_disease_risk.csv  # Raw dataset (1,200 rows)
│   └── processed/
│       ├── train.csv              # Processed training split (960 rows)
│       └── test.csv               # Processed held-out test split (240 rows)
│
├── notebooks/
│   └── 01_data_audit_eda.ipynb    # Executable Jupyter Notebook for EDA & Data Audit
│
├── src/
│   ├── __init__.py
│   ├── data_validation.py         # Schema & input range validation
│   ├── preprocessing.py          # ColumnTransformer & Pipeline builder
│   ├── modeling.py               # Candidate models & GridSearchCV tuning
│   ├── metrics.py                # Classification evaluation metrics
│   ├── explainability.py         # Feature importances & What-If simulator
│   ├── inference.py              # Real-time single observation prediction
│   ├── generate_dataset.py       # Benchmark dataset generator script
│   └── run_eda_visuals.py        # Static figure generator script
│
├── artifacts/
│   ├── model_pipeline.joblib     # Serialized fitted Scikit-Learn pipeline
│   └── model_metadata.json       # Model parameters & test evaluation metrics
│
├── reports/
│   ├── figures/                  # Saved EDA & evaluation plots
│   ├── metrics.json              # Detailed test performance metrics
│   ├── technical_paper.md        # Comprehensive 12-section technical paper
│   └── viva_guide.md             # Student presentation & viva defense guide
│
├── tests/
│   ├── test_data_validation.py   # Schema & range unit tests
│   ├── test_preprocessing.py     # Pipeline transformation unit tests
│   ├── test_prediction.py        # Inference pipeline integration tests
│   └── test_artifact.py          # Artifact integrity unit tests
│
└── assets/
    └── project_logo.png           # Custom visual asset logo
```

---

## 4. Measured Experimental Results

### Candidate Model 5-Fold Cross-Validation Comparison

| Model Candidate | CV Mean Macro F1 | CV Std Dev | Selected Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Random Forest (Winner)** | **0.7368** | **0.0245** | `max_depth=8, n_estimators=50` |
| Decision Tree | 0.7179 | 0.0190 | `max_depth=5, min_samples_split=5` |
| Logistic Regression | 0.7042 | 0.0395 | `C=0.1, solver='lbfgs'` |
| K-Nearest Neighbors | 0.6807 | 0.0321 | `n_neighbors=9, weights='distance'` |
| Dummy Classifier | 0.1728 | 0.0008 | Default (Most Frequent) |

### Final Held-Out Test Set Metrics (n=240)

* **Accuracy:** 72.92%
* **Balanced Accuracy:** 72.71%
* **Macro Precision:** 0.7227
* **Macro Recall:** 0.7271
* **Macro F1-Score:** **0.7240**
* **Weighted F1-Score:** 0.7255

---

## 5. Execution Commands & Setup Guide

### Step 1: Virtual Environment Setup & Dependencies

```bash
# Create Python virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (macOS / Linux)
# source .venv/bin/activate

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Generate Dataset & Static EDA Figures

```bash
python src/generate_dataset.py
python src/run_eda_visuals.py
python src/create_eda_notebook.py
```

### Step 3: Run Model Training & Evaluation

```bash
# Train candidate models, perform 5-fold CV, and save pipeline artifact
python train.py

# Evaluate saved pipeline on held-out test set & generate figures
python evaluate.py
```

### Step 4: Run Automated Pytest Suite

```bash
python -m pytest -v
```

### Step 5: Launch Streamlit Web Application

```bash
streamlit run app.py
```

---

## 6. Streamlit Dashboard Features

The application shell consists of 5 complete pages:
1. **Overview Dashboard:** Live dataset statistics, summary cards, risk distribution donut chart, system architecture flow, and disclaimers.
2. **Predict Crop Disease Risk:** Organized form input controls, dynamic domain validation, risk category badges, model confidence score distribution, driving factor explanations, and interactive **What-If Scenario Simulator**.
3. **Data Analytics:** Interactive Plotly histograms, boxplots, correlation heatmaps, category filters, and written analytical findings.
4. **Model Insights:** Cross-validation model comparison tables, test set confusion matrix, per-class performance bar chart, and global Gini feature importances.
5. **About Project:** Academic methodology, pipeline architecture diagram, tech stack details, limitations, and literature references.

---

## 7. Educational & Responsible Use Disclaimer

> **IMPORTANT:** This application is built as an academic capstone prototype. The predicted risk categories reflect statistical associations in dataset observations and do NOT provide a certified phytopathological laboratory diagnosis or chemical pesticide recommendation. Field decisions should always involve qualified agricultural extension services.
