import streamlit as st\n
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

import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("style.css not found. Glassmorphic styles might be missing.")

# Load Model and Scaler
@st.cache_resource
def load_model():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.joblib.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Model or Scaler not found. Please run 'setup_model.py' first.")
    st.stop()

# Function to map model clusters to User-Friendly IDs (0-4)
def get_cluster_mapping(kmeans_model, scaler):
    """
    Analyzes centroids to create a mapping from Model Cluster ID -> User Friendly ID (0-4).
    Target Mapping based on User Request:
    0: Low Income, Low Spending (Sensible)
    1: Low Income, High Spending (Careless)
    2: Medium Income, Medium Spending (Standard)
    3: High Income, Low Spending (Frugal)
    4: High Income, High Spending (Elite)
    """
    centers_scaled = kmeans_model.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    
    # Create a DataFrame for easy logic
    df_centers = pd.DataFrame(centers, columns=['Income', 'Score'])
    df_centers['model_id'] = range(len(df_centers))
    
    mapping = {}
    
    # 1. Standard (Medium Income, Medium Score) - usually the middle cluster
    # Closest to mean (approx 60, 50)
    # We can find the one with Income closest to mean income of centroids
    # But explicitly: 
    # Standard: Income ~40-70, Score ~40-60
    
    # Let's sort by Income first
    df_sorted = df_centers.sort_values('Income').reset_index(drop=True)
    
    # The two lowest incomes are "Low Income" group
    low_income = df_sorted.iloc[:2] 
    # The lowest Score in this group is Group 0 (Low, Low)
    id_0 = low_income.sort_values('Score').iloc[0]['model_id']
    # The highest Score in this group is Group 1 (Low, High) -> "Careless"
    id_1 = low_income.sort_values('Score').iloc[1]['model_id']
    
    # The two highest incomes are "High Income" group
    high_income = df_sorted.iloc[-2:]
    # The lowest Score in this group is Group 3 (High, Low) -> "Frugal"
    id_3 = high_income.sort_values('Score').iloc[0]['model_id']
    # The highest Score in this group is Group 4 (High, High) -> "Elite"
    id_4 = high_income.sort_values('Score').iloc[1]['model_id']
    
    # The remaining one is Standard
    remaining_ids = set(df_centers['model_id']) - {id_0, id_1, id_3, id_4}
    id_2 = list(remaining_ids)[0]
    
    mapping = {
        int(id_0): {"label": "Sensible Customer", "id": 0, "desc": "Low Income, Low Spending. Focus on value and discounts."},
        int(id_1): {"label": "Careless Spender", "id": 1, "desc": "Low Income, High Spending. Target w/ impulse deals (Caution: Credit Risk)."},
        int(id_2): {"label": "Standard Customer", "id": 2, "desc": "Average Income, Average Spending. Maintain engagement."},
        int(id_3): {"label": "Frugal / Target", "id": 3, "desc": "High Income, Low Spending. Upsell value proposition."},
        int(id_4): {"label": "Elite / Target", "id": 4, "desc": "High Income, High Spending. Premium offers and loyalty programs."}
    }
    
    return mapping

cluster_mapping = get_cluster_mapping(model, scaler)

# Main Page - Title and Description Centered
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="margin-bottom: 1rem;">🛍️ Customer Segmentation System</h1>
    <p style="font-size: 1.1rem; color: #e0e0e0;">
        This application uses <strong>K-Means Clustering</strong> to group customers based on their <strong>Annual Income</strong> and <strong>Spending Score</strong>.<br>
        Enter customer data below to detect their segment (Groups 0-4).
    </p>
</div>
""", unsafe_allow_html=True)

# Input Section - Centered without box
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50, step=1, key="income")
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1, key="score")
    
    # Predict button centered below inputs
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict Segment", use_container_width=True)

# Prediction Logic
if predict_button:
    # Prepare input
    input_data = np.array([[annual_income, spending_score]])
    
    # Scale input
    scaled_data = scaler.transform(input_data)
    
    # Predict
    model_cluster_id = model.predict(scaled_data)[0]
    
    # Get mapped info
    info = cluster_mapping.get(int(model_cluster_id))
    user_id = info['id']
    label = info['label']
    advice = info['desc']
    
    # Color logic with enhanced gradients
    color = "#00d2ff"
    gradient = "linear-gradient(135deg, #00d2ff, #0099cc)"
    emoji = "🎯"
    
    if user_id == 4: 
        color = "#ffd700"
        gradient = "linear-gradient(135deg, #ffd700, #ffA500)"
        emoji = "👑"
    elif user_id == 1: 
        color = "#ff6b6b"
        gradient = "linear-gradient(135deg, #ff6b6b, #ee5a6f)"
        emoji = "⚠️"
    elif user_id == 3:
        color = "#00ff88"
        gradient = "linear-gradient(135deg, #00ff88, #00cc6a)"
        emoji = "💎"
    elif user_id == 2:
        color = "#5b8fb9"
        gradient = "linear-gradient(135deg, #5b8fb9, #4a7a9e)"
        emoji = "⭐"
    elif user_id == 0:
        color = "#a0a0ff"
        gradient = "linear-gradient(135deg, #a0a0ff, #7070dd)"
        emoji = "🎯"
    
    # Calculate RGB values for shadow effects
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    
    # Display Results with stunning effects
    st.markdown(f"""
<div class="result-card glitter-border" style="border: 2px solid {color}; box-shadow: 0 0 30px rgba({r}, {g}, {b}, 0.4), 0 0 60px rgba({r}, {g}, {b}, 0.2), inset 0 0 80px rgba({r}, {g}, {b}, 0.1);">
    <div class="result-header" style="background: {gradient};">
        <h2 class="pulse-glow">{emoji} Group {user_id}: {label}</h2>
    </div>
    <div class="result-content">
        <div class="profile-section">
            <h3 style="color: {color}; text-shadow: 0 0 20px {color};">💡 Customer Profile</h3>
            <p class="profile-text">{advice}</p>
        </div>
        <div class="input-data-card" style="border-left: 4px solid {color}; background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.2));">
            <div class="data-row">
                <span class="data-label">💰 Annual Income:</span>
                <span class="data-value" style="color: {color};">${annual_income}k</span>
            </div>
            <div class="data-row">
                <span class="data-label">📊 Spending Score:</span>
                <span class="data-value" style="color: {color};">{spending_score}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Explanation Section
    with st.expander("ℹ️ Why this group? (See Centroid Distances)"):
        st.write("The model assigns the group based on which cluster center (Centroid) is closest to your input.")
        
        # Calculate distances to all centroids
        distances = model.transform(scaled_data)[0]
        
        # Create a nice dataframe for display
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        explanation_data = []
        
        for mid, dist in enumerate(distances):
            # Get the mapped user ID for this model ID
            map_info = cluster_mapping.get(mid)
            uid = map_info['id']
            ulabel = map_info['label']
            
            centroid_income = centers_original[mid][0]
            centroid_score = centers_original[mid][1]
            
            is_selected = (mid == model_cluster_id)
            status = "✅ Selected (Closest)" if is_selected else ""
            
            explanation_data.append({
                "User Group": f"Group {uid}: {ulabel}",
                "Centroid Income ($k)": f"{centroid_income:.1f}",
                "Centroid Score": f"{centroid_score:.1f}",
                "Distance": f"{dist:.4f}",
                "Status": status
            })
            
        df_explain = pd.DataFrame(explanation_data).sort_values("Distance")
        st.dataframe(df_explain, use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; opacity: 0.6;">
    <p>© 2026 Customer Segmentation Project | Partha Sarathi R</p>
</div>
""", unsafe_allow_html=True)
\n\nst.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)\n