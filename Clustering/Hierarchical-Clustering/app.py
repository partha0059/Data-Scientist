import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
\n# ==========================================
# Hierarchical Clustering - Iris Segmentation
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
    page_title="Iris Clustering App",
    page_icon="�",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
    <style>
    h1 {
        color: black;
        text-align: center;
    }
    .stButton>button {
        background-color: black;
        color: white;
        height: 50px;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Iris Flower Segmentation")
st.write("Hierarchical Clustering using Agglomerative Method")

# -----------------------------
# Load Scaler
# -----------------------------
scaler = joblib.load("iris_scaler.pkl")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", value=3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm)", value=1.4)
    petal_width = st.number_input("Petal Width (cm)", value=0.2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Cluster"):

    input_data = np.array([[sepal_length,
                            sepal_width,
                            petal_length,
                            petal_width]])

    scaled_input = scaler.transform(input_data)

    centroids = np.load("cluster_centroids.npy")
    distances = np.linalg.norm(centroids - scaled_input, axis=1)
    cluster = int(np.argmin(distances))

    st.subheader("Cluster Assignment:")
    st.success(f"Cluster {cluster}")

    # Interpretation
    if cluster == 0:
        st.info("Likely resembles Setosa-type cluster characteristics")
    elif cluster == 1:
        st.warning("Likely resembles Versicolor-type cluster characteristics")
    else:
        st.error("Likely resembles Virginica-type cluster characteristics")

st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
