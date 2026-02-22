# ==========================================
# KNN Breast Cancer Prediction - Streamlit
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

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f7f9;
    }
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #c0392b;
        text-align: center;
    }
    .stButton>button {
        background-color: blue;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Classification using KNN")

st.write("Enter patient diagnostic values below:")

# ----------------------------
# Load Model & Scaler
# ----------------------------
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Input Fields (30 Features)
# ----------------------------

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

inputs = []

col1, col2 = st.columns(2)

for i in range(15):
    value = col1.number_input(feature_names[i], value=0.0)
    inputs.append(value)

for i in range(15, 30):
    value = col2.number_input(feature_names[i], value=0.0)
    inputs.append(value)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Diagnosis"):

    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.success("✅ Prediction: Benign (Non-Cancerous)")
    else:
        st.error("⚠️ Prediction: Malignant (Cancerous)")

st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
