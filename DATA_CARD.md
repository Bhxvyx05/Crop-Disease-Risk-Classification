# DATA CARD: Crop Disease Risk Dataset

## 1. Dataset Overview

* **Dataset Name:** Crop Disease Risk Benchmark Dataset
* **Domain:** Agriculture / Phytopathology / Environmental Epidemiology
* **Task:** Multi-Class Supervised Classification (`Low`, `Moderate`, `High` disease risk)
* **Dataset Format:** CSV (Comma-Separated Values)
* **Total Observations:** 1,200 records
* **Total Input Features:** 7 features (5 numerical, 2 categorical)
* **Target Variable:** `disease_risk`
* **License:** Open Academic / Educational Use Only

---

## 2. Target Variable & Label Semantics

| Target Label | Class Code | Agronomic Definition | Environmental Profile |
| :--- | :--- | :--- | :--- |
| **Low** | 0 | Unfavorable environmental conditions for spore germination and infection. Disease incidence probability is minimal. | Low humidity (<50%), short leaf wetness (<4h), cool or very dry weather. |
| **Moderate** | 1 | Favorable micro-environmental conditions for opportunistic pathogen development. Monitoring recommended. | Intermediate humidity (55-75%), moderate leaf wetness (5-10h), temperate range. |
| **High** | 2 | Optimal epidemic micro-climate. High risk of fungal/bacterial proliferation (e.g. Blight, Mildew, Rust). Immediate intervention needed. | High humidity (>75%), prolonged leaf wetness (>10h), optimal temperature (18-30°C), heavy rainfall. |

---

## 3. Input Feature Dictionary

### Numerical Features

| Feature Name | Data Type | Range / Bounds | Unit | Description | Agronomic Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature_c` | Float | 10.0 to 45.0 | °C | Ambient air temperature | Influences pathogen metabolic activity & spore germination rates. |
| `humidity_pct` | Float | 20.0 to 100.0 | % | Relative air humidity | Crucial factor for fungal hyphae elongation & bacterial motility. |
| `soil_moisture_pct` | Float | 5.0 to 90.0 | % | Volumetric soil moisture | Affects root zone water availability and soil-borne fungal pathogens. |
| `rainfall_mm` | Float | 0.0 to 200.0 | mm | Recent precipitation (24h) | Promotes splash dispersal of spores and micro-climate humidification. |
| `leaf_wetness_hours` | Float | 0.0 to 24.0 | hours/day | Daily leaf wetness duration | Primary physical prerequisite for leaf-cuticle penetration by spores. |

### Categorical Features

| Feature Name | Data Type | Categories / Allowed Values | Description | Agronomic Role |
| :--- | :--- | :--- | :--- | :--- |
| `crop_type` | String | `Wheat`, `Rice`, `Maize`, `Tomato`, `Potato` | Crop species under observation | Differential genetic resistance & pathogen host specificity. |
| `growth_stage` | String | `Seedling`, `Vegetative`, `Flowering`, `Maturity` | Physiological growth stage | Flowering and vegetative stages exhibit higher physiological susceptibility. |

---

## 4. Data Quality & Data Audit Summary

* **Missing Values:** < 1.0% artificial missingness present in `temperature_c` and `humidity_pct` for testing automated median imputation pipelines.
* **Duplicate Rows:** 0 exact duplicate rows detected.
* **Outliers:** Natural environmental extreme values (e.g., rainfall > 150mm, temperature > 40°C) retained as valid domain extremes.
* **Data Leakage Check:** Target `disease_risk` is strictly evaluated after feature measurements. All preprocessing steps (imputation, scaling, one-hot encoding) are fitted exclusively on training splits during cross-validation.

---

## 5. Intended Use & Educational Disclaimer

> **IMPORTANT DISCLAIMER:**
> This dataset and associated predictive models are created strictly for **educational and academic research prototype purposes**. The model estimates risk categories based on environmental indicators and does NOT provide a laboratory-confirmed phytopathological diagnosis or chemical treatment prescription. Real-world agricultural decisions should be guided by qualified local agronomic experts and certified extension agents.
