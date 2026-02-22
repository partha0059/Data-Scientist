import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
\n# ==========================================
# Feature Selection Comparison App
# Breast Cancer Dataset
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
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Feature Selection Comparison",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Feature Selection: Filter vs Wrapper")
st.write("Comparison using Breast Cancer Dataset")

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# -----------------------------
# Load Models & Objects
# -----------------------------
baseline_model = joblib.load("baseline_model.pkl")
filter_model = joblib.load("filter_model.pkl")
wrapper_model = joblib.load("wrapper_model.pkl")

scaler = joblib.load("scaler.pkl")
filter_selector = joblib.load("filter_selector.pkl")
wrapper_selector = joblib.load("wrapper_selector.pkl")

# -----------------------------
# Accuracy Calculation
# -----------------------------
X_scaled = scaler.transform(X)

baseline_acc = accuracy_score(y, baseline_model.predict(X_scaled))
filter_acc = accuracy_score(
    y,
    filter_model.predict(filter_selector.transform(X_scaled))
)
wrapper_acc = accuracy_score(
    y,
    wrapper_model.predict(wrapper_selector.transform(X_scaled))
)

st.subheader("Model Accuracy Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Baseline (All Features)", round(baseline_acc, 4))
col2.metric("Filter Method (SelectKBest)", round(filter_acc, 4))
col3.metric("Wrapper Method (RFE)", round(wrapper_acc, 4))

st.divider()

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Try Prediction")

method = st.selectbox(
    "Choose Model",
    ["Baseline", "Filter Method", "Wrapper Method"]
)

inputs = []

st.write("Enter first 10 feature values (for demo):")

for i in range(10):
    value = st.number_input(feature_names[i], value=0.0)
    inputs.append(value)

if st.button("Predict"):

    input_array = np.array([inputs])

    # Pad to full 30 features for baseline
    if method == "Baseline":
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        prediction = baseline_model.predict(scaled)

    elif method == "Filter Method":
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        selected = filter_selector.transform(scaled)
        prediction = filter_model.predict(selected)

    else:
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        selected = wrapper_selector.transform(scaled)
        prediction = wrapper_model.predict(selected)

    if prediction[0] == 1:
        st.success("✅ Benign Tumor")
    else:
        st.error("⚠️ Malignant Tumor")

st.divider()

# -----------------------------
# Show Selected Features
# -----------------------------
st.subheader("Selected Features")

filter_features = feature_names[filter_selector.get_support()]
wrapper_features = feature_names[wrapper_selector.get_support()]

col4, col5 = st.columns(2)

with col4:
    st.write("### Filter Method Features")
    for f in filter_features:
        st.write("-", f)

with col5:
    st.write("### Wrapper Method Features")
    for f in wrapper_features:
        st.write("-", f)

st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
