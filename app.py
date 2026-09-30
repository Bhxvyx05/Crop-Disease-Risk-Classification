import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Keep Plotly figures visually consistent with the dark dashboard theme.
px.defaults.template = "plotly_dark"

from src.data_validation import validate_single_observation, load_config
from src.inference import ModelInferencePipeline
from src.explainability import get_global_feature_importance, simulate_what_if

# Set page configuration
st.set_page_config(
    page_title="Crop Disease Risk AI | Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern SaaS UI/UX Design System
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --app-bg: #101827;
        --sidebar-bg: #142A27;
        --card-bg: #1B2738;
        --card-border: #2D3B4E;
        --text-main: #F1F5F9;
        --text-muted: #A6B4C8;
        --green: #4DAA63;
        --green-dark: #24543B;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    section.main, .main {
        background-color: var(--app-bg) !important;
        color: var(--text-main) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(16, 24, 39, 0.96) !important;
    }

    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background: transparent !important;
    }

    /* Sidebar: dark background and readable text */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background-color: var(--sidebar-bg) !important;
        color: var(--text-main) !important;
        border-right: 1px solid #29443A;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border-radius: 9px;
        padding: 0.45rem 0.55rem;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #203D31 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--green-dark) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #17472C 0%, #247A42 100%);
        color: #FFFFFF;
        padding: 2rem 2.5rem;
        border: 1px solid #2D7544;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: #FFFFFF !important;
    }

    .app-header p {
        font-size: 1.05rem;
        color: #E1F3E5 !important;
        margin: 0;
    }

    /* Dark metric cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.25rem 1.2rem;
        min-height: 112px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.12);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4B8060;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }

    .metric-card .title {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.45rem;
    }

    .metric-card .value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F1F5F9;
        overflow-wrap: anywhere;
    }

    .metric-card .subtitle {
        font-size: 0.78rem;
        color: #78C58D;
        margin-top: 0.3rem;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        letter-spacing: 0.03em;
    }

    .risk-badge-low {
        background-color: #173B29;
        color: #9BE0AD;
        border: 1.5px solid #4DAA63;
    }

    .risk-badge-moderate {
        background-color: #49331A;
        color: #FFD08A;
        border: 1.5px solid #D99536;
    }

    .risk-badge-high {
        background-color: #4A2228;
        color: #FFB1B1;
        border: 1.5px solid #D84B4B;
    }

    /* Prediction result card */
    .prediction-result-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: var(--text-main);
    }

    .prediction-result-card h4 {
        margin-bottom: 0.8rem;
        color: var(--text-muted) !important;
    }

    .prediction-result-card p {
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #D6DFEA !important;
    }

    /* Disclaimer */
    .disclaimer-box {
        background-color: #192B40;
        border-left: 4px solid #5799E8;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #D5E6FF;
        margin-top: 1.5rem;
    }

    /* General Streamlit surfaces and text */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stText"], .stCaption, label {
        color: var(--text-main);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #F1F5F9 !important;
    }

    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 0.85rem;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: var(--text-main) !important;
    }

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {
        background-color: #172235 !important;
        color: #F1F5F9 !important;
        border-color: #3A4A60 !important;
    }

    [data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg {
        fill: #DCE5EF !important;
    }

    [data-testid="stForm"], [data-testid="stExpander"] {
        background-color: #172235;
        border: 1px solid var(--card-border);
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--card-border);
        border-radius: 8px;
        overflow: hidden;
    }

    button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        background: #247A42 !important;
        color: #FFFFFF !important;
        border: 1px solid #4DAA63 !important;
        border-radius: 8px !important;
    }

    button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {
        background: #2E9251 !important;
        border-color: #78C58D !important;
    }

    hr {
        border-color: #2D3B4E !important;
    }

    @media (max-width: 900px) {
        .app-header { padding: 1.4rem; }
        .app-header h1 { font-size: 1.7rem; }
        .metric-card { padding: 1rem; }
        .metric-card .value { font-size: 1.35rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper functions for caching
@st.cache_resource
def load_app_config():
    return load_config("config/config.yaml")

@st.cache_resource
def load_inference_pipeline():
    try:
        return ModelInferencePipeline()
    except Exception as e:
        st.error(f"Error loading model inference pipeline: {e}")
        return None

@st.cache_data
def load_dataset_cached():
    return pd.read_csv("data/raw/crop_disease_risk.csv")

@st.cache_data
def load_metadata_cached():
    if os.path.exists("artifacts/model_metadata.json"):
        with open("artifacts/model_metadata.json", "r") as f:
            return json.load(f)
    return {}

config = load_app_config()
inf_pipeline = load_inference_pipeline()
df_raw = load_dataset_cached()
metadata = load_metadata_cached()

# Sidebar Navigation Setup
with st.sidebar:
    st.image("assets/project_logo.png", width=70)
    st.markdown("### Crop Risk Classification")
    st.caption("AI-Powered Agricultural Decision Support")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        options=[
            "Overview Dashboard",
            "Predict Crop Disease Risk",
            "Data Analytics",
            "Model Insights",
            "About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### Model Status")
    if metadata:
        st.success(f"**Model:** {metadata.get('model_name', 'Fitted Model')}")
        st.info(f"**Test Accuracy:** {metadata.get('test_metrics', {}).get('accuracy', 0.0):.1%}")
        st.info(f"**Macro F1:** {metadata.get('test_metrics', {}).get('macro_f1', 0.0):.3f}")
    else:
        st.warning("Model metadata pending.")
        
    st.markdown("---")
    st.caption("Capstone Project v1.0 | 2026")
    st.caption("Free & Open Source ML Stack")

# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if page == "Overview Dashboard":
    st.markdown(
        """
        <div class="app-header">
            <h1>Crop Disease Risk Classification</h1>
            <p>AI-powered micro-climate & environmental assessment for agricultural phytopathology decision support.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Live Dataset & Model Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Records</div>
                <div class="value">{len(df_raw)}</div>
                <div class="subtitle">Sample observations</div>
            </div>""", unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Features</div>
                <div class="value">7</div>
                <div class="subtitle">5 Env + 2 Crop</div>
            </div>""", unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Classes</div>
                <div class="value">3</div>
                <div class="subtitle">Low / Mod / High</div>
            </div>""", unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Selected ML</div>
                <div class="value" style="font-size:1.3rem;">{metadata.get('model_name', 'RandomForest')}</div>
                <div class="subtitle">Best CV candidate</div>
            </div>""", unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Test Accuracy</div>
                <div class="value">{metadata.get('test_metrics', {}).get('accuracy', 0.0):.1%}</div>
                <div class="subtitle">Held-out test set</div>
            </div>""", unsafe_allow_html=True
        )
    with c6:
        st.markdown(
            f"""<div class="metric-card">
                <div class="title">Macro F1</div>
                <div class="value">{metadata.get('test_metrics', {}).get('macro_f1', 0.0):.3f}</div>
                <div class="subtitle">Class-balanced F1</div>
            </div>""", unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Summaries
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Disease Risk Category Distribution")
        target_counts = df_raw['disease_risk'].value_counts().reset_index()
        target_counts.columns = ['Risk Category', 'Count']
        fig_donut = px.pie(
            target_counts, values='Count', names='Risk Category', hole=0.45,
            color='Risk Category',
            color_discrete_map={'Low': '#2E7D32', 'Moderate': '#F57C00', 'High': '#D32F2F'}
        )
        fig_donut.update_traces(textinfo='percent+label', textfont_size=13)
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_right:
        st.subheader("Key Environmental Indicators Overview")
        df_summary = df_raw[['temperature_c', 'humidity_pct', 'soil_moisture_pct', 'leaf_wetness_hours']].mean().reset_index()
        df_summary.columns = ['Feature', 'Dataset Mean']
        fig_bar = px.bar(
            df_summary, x='Feature', y='Dataset Mean',
            color='Feature', color_discrete_sequence=['#1E4620', '#2E7D32', '#4CAF50', '#81C784']
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.subheader("How It Works")
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.markdown("#### 1. Input Observations")
        st.write("Enter air temperature, humidity, soil moisture, rainfall, leaf wetness, crop variety, and growth stage.")
    with step2:
        st.markdown("#### 2. Machine Learning Pipeline")
        st.write("Scikit-learn pipeline applies median imputation, scaling, and one-hot encoding without data leakage.")
    with step3:
        st.markdown("#### 3. Risk Classification")
        st.write("The validated machine learning model evaluates pathogen micro-climate suitability (Low, Moderate, High).")
    with step4:
        st.markdown("#### 4. Actionable Insights")
        st.write("Review key driving factors, confidence metrics, and simulate What-If parameter modifications.")
        
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Educational Prototype Disclaimer:</strong> This application is developed strictly for academic evaluation and research prototyping. The predicted disease-risk categories reflect dataset-learned statistical patterns and do NOT constitute a certified plant pathology diagnosis or chemical treatment advice.
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# PAGE 2: PREDICT CROP DISEASE RISK
# ==============================================================================
elif page == "Predict Crop Disease Risk":
    st.subheader("Predict Crop Disease Risk Category")
    st.caption("Enter current environmental metrics and crop parameters to obtain model-based risk predictions.")
    
    # Form Input Controls
    with st.form("prediction_form"):
        col_env, col_crop = st.columns(2)
        
        with col_env:
            st.markdown("#### Environmental Conditions")
            temp = st.number_input("Air Temperature (°C)", min_value=10.0, max_value=45.0, value=26.5, step=0.5, help="Valid range: 10.0 - 45.0 °C")
            humidity = st.number_input("Relative Humidity (%)", min_value=20.0, max_value=100.0, value=82.0, step=1.0, help="Valid range: 20.0 - 100.0 %")
            soil_moisture = st.number_input("Soil Moisture (%)", min_value=5.0, max_value=90.0, value=55.0, step=1.0, help="Valid range: 5.0 - 90.0 %")
            rainfall = st.number_input("Recent Rainfall (mm)", min_value=0.0, max_value=200.0, value=45.0, step=2.0, help="Valid range: 0.0 - 200.0 mm")
            leaf_wetness = st.number_input("Leaf Wetness (hours/day)", min_value=0.0, max_value=24.0, value=11.5, step=0.5, help="Valid range: 0.0 - 24.0 hrs/day")
            
        with col_crop:
            st.markdown("#### Crop Information")
            crop = st.selectbox("Crop Type", options=config['features']['categorical']['crop_type']['categories'], index=3) # Tomato
            growth = st.selectbox("Growth Stage", options=config['features']['categorical']['growth_stage']['categories'], index=2) # Flowering
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Predict Disease Risk", type="primary", use_container_width=True)
            
    if submit_btn:
        input_data = {
            'temperature_c': temp,
            'humidity_pct': humidity,
            'soil_moisture_pct': soil_moisture,
            'rainfall_mm': rainfall,
            'leaf_wetness_hours': leaf_wetness,
            'crop_type': crop,
            'growth_stage': growth
        }
        
        with st.spinner("Processing observation through ML Pipeline..."):
            res = inf_pipeline.predict_single(input_data)
            
        if not res['success']:
            st.error("Input Validation Errors Detected:")
            for err in res['errors']:
                st.write(f"- {err}")
        else:
            pred_class = res['prediction']
            st.markdown("---")
            st.subheader("Prediction Results & Risk Analysis")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                badge_class = f"risk-badge risk-badge-{pred_class.lower()}"
                st.markdown(
                    f"""
                    <div class="prediction-result-card">
                        <h4>Predicted Disease Risk Category</h4>
                        <div class="{badge_class}">{pred_class.upper()} RISK</div>
                        <p>
                            {'High micro-climatic pathogen pressure detected. Immediate preventive crop monitoring recommended.' if pred_class == 'High' else ('Moderate pathogen micro-climate. Routine field surveillance advised.' if pred_class == 'Moderate' else 'Environmental conditions are currently unfavorable for epidemic disease proliferation.')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with res_col2:
                st.markdown("#### Model-Estimated Class Scores")
                if res['class_probabilities']:
                    prob_df = pd.DataFrame([res['class_probabilities']]).T.reset_index()
                    prob_df.columns = ['Risk Class', 'Model Score']
                    fig_prob = px.bar(
                        prob_df, x='Model Score', y='Risk Class', orientation='h',
                        color='Risk Class', color_discrete_map={'Low': '#2E7D32', 'Moderate': '#F57C00', 'High': '#D32F2F'},
                        text_auto='.1%'
                    )
                    fig_prob.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    st.caption("*Note: Model-estimated class scores reflect statistical confidence on learned dataset samples, not calibrated field probabilities.*")
                    
            st.markdown("---")
            st.markdown("#### Explanation Panel: Key Driving Factors")
            for factor in res['explanation']['agronomic_factors']:
                st.markdown(f"• **{factor}**")
                
            # What-If Simulator Container
            st.markdown("---")
            st.subheader("What-If Scenario Simulator")
            st.caption("Modify an environmental parameter below to observe how the predicted risk category changes.")
            
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                mod_feat = st.selectbox("Select Parameter to Modify", options=['leaf_wetness_hours', 'humidity_pct', 'temperature_c', 'rainfall_mm'])
                if mod_feat == 'leaf_wetness_hours':
                    new_val = st.slider("New Leaf Wetness (hrs)", 0.0, 24.0, float(leaf_wetness))
                elif mod_feat == 'humidity_pct':
                    new_val = st.slider("New Humidity (%)", 20.0, 100.0, float(humidity))
                elif mod_feat == 'temperature_c':
                    new_val = st.slider("New Temperature (°C)", 10.0, 45.0, float(temp))
                else:
                    new_val = st.slider("New Rainfall (mm)", 0.0, 200.0, float(rainfall))
                    
            with sim_col2:
                base_df = pd.DataFrame([input_data])
                sim_res = simulate_what_if(inf_pipeline.pipeline, base_df, mod_feat, new_val)
                
                st.write(f"Original Prediction: **{sim_res['original_prediction']} Risk**")
                st.write(f"Simulated Prediction: **{sim_res['modified_prediction']} Risk**")
                
                if sim_res['original_prediction'] != sim_res['modified_prediction']:
                    st.success(f"Shift detected! Changing `{mod_feat}` to `{new_val}` shifts prediction from **{sim_res['original_prediction']}** to **{sim_res['modified_prediction']}** Risk.")
                else:
                    st.info(f"Modifying `{mod_feat}` to `{new_val}` preserves the **{sim_res['original_prediction']}** Risk classification.")

# ==============================================================================
# PAGE 3: DATA ANALYTICS
# ==============================================================================
elif page == "Data Analytics":
    st.subheader("Dataset Exploration & Data Analytics")
    st.caption("Interactive analysis of environmental features, class distributions, and feature correlations.")
    
    # Dataset Summary Table
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Rows", len(df_raw))
    t2.metric("Total Columns", len(df_raw.columns))
    t3.metric("Missing Cells", df_raw.isnull().sum().sum())
    t4.metric("Exact Duplicate Rows", df_raw.duplicated().sum())
    
    st.markdown("---")
    
    # Interactive Filters
    st.markdown("#### Analytics Controls & Filters")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        selected_risk = st.multiselect("Filter by Risk Category", options=['Low', 'Moderate', 'High'], default=['Low', 'Moderate', 'High'])
    with f_col2:
        selected_crop = st.multiselect("Filter by Crop Type", options=config['features']['categorical']['crop_type']['categories'], default=config['features']['categorical']['crop_type']['categories'])
        
    filtered_df = df_raw[df_raw['disease_risk'].isin(selected_risk) & df_raw['crop_type'].isin(selected_crop)]
    st.write(f"Showing **{len(filtered_df)}** of {len(df_raw)} records matching filters.")
    
    st.markdown("---")
    
    # Visual Charts Grid
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Feature Distributions", "Risk Category Comparisons", "Correlation Matrix"])
    
    with chart_tab1:
        feat_to_plot = st.selectbox("Select Numerical Feature", options=['temperature_c', 'humidity_pct', 'soil_moisture_pct', 'rainfall_mm', 'leaf_wetness_hours'])
        fig_hist = px.histogram(
            filtered_df, x=feat_to_plot, color='disease_risk', marginal='box',
            color_discrete_map={'Low': '#2E7D32', 'Moderate': '#F57C00', 'High': '#D32F2F'},
            barmode='overlay', opacity=0.7, title=f'Distribution of {feat_to_plot} by Disease Risk'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with chart_tab2:
        fig_box = px.box(
            filtered_df, x='disease_risk', y=['leaf_wetness_hours', 'humidity_pct', 'temperature_c'],
            category_orders={'disease_risk': ['Low', 'Moderate', 'High']},
            title="Environmental Indicators Comparison Across Disease Risk Levels"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with chart_tab3:
        num_cols = ['temperature_c', 'humidity_pct', 'soil_moisture_pct', 'rainfall_mm', 'leaf_wetness_hours']
        corr_matrix = filtered_df[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto='.2f', aspect='auto',
            color_continuous_scale='YlGnBu', title="Feature Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Analytical Observations")
    st.markdown("""
    - **Leaf Wetness Duration:** Strongest individual discriminator. Observations exceeding 10 hours/day predominantly fall into the **High Risk** category.
    - **Relative Humidity:** High relative humidity (>75%) coupled with moderate temperatures (18-30°C) creates peak epidemic risk clusters.
    - **Crop Specificity:** Crop varieties show uniform distribution across risk levels due to controlled environmental simulation.
    """)

# ==============================================================================
# PAGE 4: MODEL INSIGHTS
# ==============================================================================
elif page == "Model Insights":
    st.subheader("Model Evaluation, Comparison & Explainability")
    st.caption("Rigorous cross-validation comparison, held-out test set performance, confusion matrix, and feature importances.")
    
    st.markdown("#### 1. Candidate Model Cross-Validation Comparison")
    if 'cv_results' in metadata:
        cv_df = pd.DataFrame(metadata['cv_results'])
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.warning("CV results metadata not found.")
        
    st.markdown("---")
    
    st.markdown("#### 2. Held-Out Test Set Performance (Final Selected Model: RandomForest)")
    if 'test_metrics' in metadata:
        tm = metadata['test_metrics']
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{tm['accuracy']:.4f}")
        m2.metric("Balanced Acc", f"{tm['balanced_accuracy']:.4f}")
        m3.metric("Macro Precision", f"{tm['macro_precision']:.4f}")
        m4.metric("Macro Recall", f"{tm['macro_recall']:.4f}")
        m5.metric("Macro F1", f"{tm['macro_f1']:.4f}")
        m6.metric("Weighted F1", f"{tm['weighted_f1']:.4f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_cm, col_imp = st.columns(2)
        
        with col_cm:
            st.subheader("Test Confusion Matrix")
            cm_data = np.array(tm['confusion_matrix'])
            fig_cm = px.imshow(
                cm_data, text_auto=True,
                x=tm['target_classes'], y=tm['target_classes'],
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_imp:
            st.subheader("Global Feature Importance")
            if inf_pipeline and inf_pipeline.pipeline:
                imp_df, _ = get_global_feature_importance(inf_pipeline.pipeline)
                fig_imp = px.bar(
                    imp_df.head(8), x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Greens'
                )
                fig_imp.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# PAGE 5: ABOUT PROJECT
# ==============================================================================
elif page == "About Project":
    st.subheader("About Crop Disease Risk Classification Capstone")
    
    st.markdown("""
    ### Project Overview
    This project is an end-to-end supervised Machine Learning application designed to assess **crop disease risk levels** from environmental and crop growth indicators. 
    By leveraging micro-climatic indicators (temperature, humidity, leaf wetness, soil moisture, precipitation), the application provides actionable risk categorization to support precision agricultural decision-making.
    
    ### System Architecture Flow
    """)
    
    st.code("""
    Raw Data (CSV) ➔ Data Schema Audit ➔ EDA & Preprocessing Pipeline ➔ 5-Fold Stratified CV
                                                                               │
    Streamlit Web App  Real-Time Inference  Serialized Joblib Artifact  Model Selection (RandomForest)
    """, language="text")
    
    st.markdown("""
    ### Technology Stack
    - **Language:** Python 3.11
    - **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, GridSearchCV)
    - **Data Science:** Pandas, NumPy
    - **Visualization:** Plotly, Matplotlib, Seaborn
    - **Application:** Streamlit (Custom CSS SaaS System)
    - **Serialization:** Joblib, JSON
    
    ### Methodological Highlights
    1. **Data Leakage Guard:** All imputation and feature scalers are encapsulated inside Scikit-Learn `Pipeline` and `ColumnTransformer` objects, ensuring fitting occurs strictly on training folds during cross-validation.
    2. **Class Imbalance Resilience:** Stratified splits and class-weighted objective functions are utilized to ensure equal importance across all disease risk tiers (`Low`, `Moderate`, `High`).
    
    ### Academic & Educational Disclaimer
    > **Disclaimer:** This software prototype is developed strictly for educational research and capstone presentation purposes. It evaluates dataset-learned risk associations and does not replace certified laboratory plant pathology diagnosis or professional agronomic field extension services.
    """)
