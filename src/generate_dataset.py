"""
Synthetic Agronomic Benchmark Dataset Generator for Crop Disease Risk Classification.
Generates realistic crop environmental observations based on epidemiological rules for plant pathogens.
"""

import os
import numpy as np
import pandas as pd
import yaml

def generate_crop_disease_dataset(n_samples=1200, random_state=42, output_path="data/raw/crop_disease_risk.csv"):
    np.random.seed(random_state)
    
    crop_types = ["Wheat", "Rice", "Maize", "Tomato", "Potato"]
    growth_stages = ["Seedling", "Vegetative", "Flowering", "Maturity"]
    
    crops = np.random.choice(crop_types, size=n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    stages = np.random.choice(growth_stages, size=n_samples, p=[0.20, 0.35, 0.30, 0.15])
    
    # Numerical environmental features
    temp = np.random.normal(loc=25.0, scale=6.0, size=n_samples)
    temp = np.clip(temp, 12.0, 42.0)
    
    humidity = np.random.uniform(low=25.0, high=98.0, size=n_samples)
    
    soil_moisture = 0.4 * humidity + np.random.normal(loc=20.0, scale=10.0, size=n_samples)
    soil_moisture = np.clip(soil_moisture, 8.0, 88.0)
    
    rainfall = np.random.exponential(scale=25.0, size=n_samples)
    rainfall = np.clip(rainfall, 0.0, 185.0)
    
    leaf_wetness = (humidity / 100.0) * 16.0 + (rainfall / 185.0) * 8.0 + np.random.normal(loc=0.0, scale=2.0, size=n_samples)
    leaf_wetness = np.clip(leaf_wetness, 0.0, 24.0)
    
    # Agronomic risk score calculation
    risk_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        score = 0.0
        # Optimal temperature range for fungal/bacterial proliferation (18C - 30C)
        if 18.0 <= temp[i] <= 32.0:
            score += 2.0
        elif temp[i] > 32.0 or temp[i] < 15.0:
            score -= 1.0
            
        # Humidity contribution
        if humidity[i] > 75.0:
            score += 3.0
        elif humidity[i] > 55.0:
            score += 1.5
            
        # Leaf wetness contribution (critical for spore germination)
        if leaf_wetness[i] > 10.0:
            score += 3.5
        elif leaf_wetness[i] > 5.0:
            score += 1.5
            
        # Rainfall & soil moisture
        if rainfall[i] > 50.0 or soil_moisture[i] > 65.0:
            score += 2.0
            
        # Susceptibility based on growth stage (Flowering & Vegetative are more vulnerable)
        if stages[i] in ["Flowering", "Vegetative"]:
            score += 1.0
            
        # Crop specific vulnerability (e.g. Tomato & Potato susceptible to blights)
        if crops[i] in ["Tomato", "Potato"]:
            score += 0.8
            
        # Add random biological noise
        score += np.random.normal(loc=0.0, scale=1.2)
        risk_scores[i] = score

    # Class assignment based on thresholds
    q33, q66 = np.percentile(risk_scores, [35, 68])
    
    disease_risk = []
    for s in risk_scores:
        if s < q33:
            disease_risk.append("Low")
        elif s < q66:
            disease_risk.append("Moderate")
        else:
            disease_risk.append("High")
            
    df = pd.DataFrame({
        'temperature_c': np.round(temp, 2),
        'humidity_pct': np.round(humidity, 2),
        'soil_moisture_pct': np.round(soil_moisture, 2),
        'rainfall_mm': np.round(rainfall, 2),
        'leaf_wetness_hours': np.round(leaf_wetness, 2),
        'crop_type': crops,
        'growth_stage': stages,
        'disease_risk': disease_risk
    })
    
    # Introduce small realistic missing values (< 1.5%) to test imputation handling
    mask_temp = np.random.rand(n_samples) < 0.01
    mask_hum = np.random.rand(n_samples) < 0.01
    df.loc[mask_temp, 'temperature_c'] = np.nan
    df.loc[mask_hum, 'humidity_pct'] = np.nan
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at '{output_path}' with shape {df.shape}.")
    return df

if __name__ == "__main__":
    generate_crop_disease_dataset()
