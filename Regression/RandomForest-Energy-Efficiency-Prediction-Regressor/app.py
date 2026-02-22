import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# Energy Efficiency - Heating Load Predictor
# Random Forest Regression
# ==========================================

import streamlit as st

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #000000);
        color: #e2e8f0;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .css-1d391kg {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    input, select, textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    .premium-footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
        font-size: 0.8rem;
        color: #64748b;
        letter-spacing: 1px;
    }
</style>
''', unsafe_allow_html=True)

import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="🏢",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #eef2f7;
    }
    h1 {
        color: #0a3d62;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669); /* Elegant Emerald Green */
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669, #047857);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span style="font-size: 32px;">🏢</span>
    <h1 style="margin: 0; color: #0a3d62;">Building Energy Efficiency Prediction</h1>
</div>
<br>
""", unsafe_allow_html=True)
st.write("Predict Heating Load using Random Forest Regression")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("energy_rf_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    relative_compactness = st.number_input("Relative Compactness", value=0.90)
    surface_area = st.number_input("Surface Area", value=600.0)
    wall_area = st.number_input("Wall Area", value=300.0)
    roof_area = st.number_input("Roof Area", value=200.0)

with col2:
    overall_height = st.selectbox("Overall Height", [3.5, 7.0])
    orientation = st.selectbox("Orientation (2-5)", [2, 3, 4, 5])
    glazing_area = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])
    glazing_area_distribution = st.selectbox("Glazing Area Distribution (0-5)", [0,1,2,3,4,5])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heating Load"):

    input_data = np.array([[relative_compactness,
                            surface_area,
                            wall_area,
                            roof_area,
                            overall_height,
                            orientation,
                            glazing_area,
                            glazing_area_distribution]])

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Heating Load:")
    st.success(f"{round(prediction, 2)} kWh/m²")

    # Simple efficiency indicator
    if prediction < 20:
        st.info("🏆 High Energy Efficiency Building")
    elif prediction < 35:
        st.warning("⚖️ Moderate Energy Efficiency")
    else:
        st.error("⚠️ Low Energy Efficiency – High Heating Demand")

st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
