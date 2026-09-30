"""
Script to execute EDA logic and generate figure artifacts in reports/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_figures():
    os.makedirs("reports/figures", exist_ok=True)
    df = pd.read_csv("data/raw/crop_disease_risk.csv")
    
    # 1. Target Distribution
    plt.figure(figsize=(7, 4.5))
    palette = {'Low': '#2E7D32', 'Moderate': '#F57C00', 'High': '#D32F2F'}
    ax = sns.countplot(data=df, x='disease_risk', order=['Low', 'Moderate', 'High'], palette=palette)
    plt.title('Target Distribution: Crop Disease Risk Categories', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Disease Risk Level', labelpad=8)
    plt.ylabel('Observation Count', labelpad=8)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df):.1%})',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/target_distribution.png', dpi=300)
    plt.close()

    # 2. Numerical Distributions
    num_cols = ['temperature_c', 'humidity_pct', 'soil_moisture_pct', 'rainfall_mm', 'leaf_wetness_hours']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#1E4620')
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('reports/figures/numerical_distributions.png', dpi=300)
    plt.close()

    # 3. Feature Boxplots by Risk Level
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    order = ['Low', 'Moderate', 'High']
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x='disease_risk', y=col, order=order, palette=palette, ax=axes[i])
        axes[i].set_title(f'{col} by Risk Category', fontweight='bold')
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('reports/figures/feature_boxplots_by_risk.png', dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Environmental Features', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=300)
    plt.close()

    # 5. Categorical Risk Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='crop_type', hue='disease_risk', hue_order=order, palette=palette, ax=ax1)
    ax1.set_title('Disease Risk by Crop Type', fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)

    sns.countplot(data=df, x='growth_stage', hue='disease_risk', hue_order=order, palette=palette, ax=ax2)
    ax2.set_title('Disease Risk by Growth Stage', fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('reports/figures/categorical_risk_breakdown.png', dpi=300)
    plt.close()

    print("All 5 EDA figures saved into reports/figures/")

if __name__ == "__main__":
    generate_eda_figures()
