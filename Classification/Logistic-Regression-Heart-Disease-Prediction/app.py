import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
🫀 Heart Disease Prediction System
Professional Medical Dashboard
Created by: Partha Sarathi R
"""

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

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Heart Disease Prediction | Partha Sarathi R",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# PROFESSIONAL CSS STYLING
# =============================================
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    
    /* Professional Clean Background */
    .stApp {
        background-color: #f9fafb;
    }
    
    /* Sidebar with Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(30, 64, 175, 0.15);
    }
    
    /* Headers - Clean Dark Text */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    /* Custom Header with Professional Clean Look */
    .dashboard-header {
        background: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-title {
        font-size: 2.25rem;
        color: #1f2937;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .creator-tag {
        background-color: #f3f4f6;
        padding: 0.6rem 1.25rem;
        border-radius: 8px;
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
    }
    
    /* Input Cards with Glassmorphism */
    .input-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid rgba(8, 145, 178, 0.15);
        box-shadow: 0 4px 16px 0 rgba(8, 145, 178, 0.08),
                    0 0 0 1px rgba(255, 255, 255, 0.4) inset;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .input-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px 0 rgba(8, 145, 178, 0.12),
                    0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        border: 1px solid rgba(8, 145, 178, 0.25);
    }
    
    .section-title {
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
        border-bottom: 2px solid #f3f4f6;
        padding-bottom: 0.6rem;
    }
    
    /* Custom Button with Professional Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1rem;
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e3a8a 0%, #0e7490 100%);
        box-shadow: 0 8px 24px rgba(30, 64, 175, 0.4);
        transform: translateY(-2px);
    }
    
    /* Result Section with Enhanced Styling */
    .report-container {
        border-top: 4px solid;
        border-image: linear-gradient(135deg, #1e40af, #0891b2) 1;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(30, 64, 175, 0.15);
        transition: all 0.3s ease;
    }
    
    .report-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 48px rgba(30, 64, 175, 0.2);
    }
    
    .risk-label {
        font-size: 0.9rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }
    
    .risk-value-low { color: #059669; font-size: 2.5rem; font-weight: 800; }
    .risk-value-med { color: #d97706; font-size: 2.5rem; font-weight: 800; }
    .risk-value-high { color: #e11d48; font-size: 2.5rem; font-weight: 800; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Adjust Sliders */
    div[class*="stSlider"] > label {
        color: #475569;
        font-weight: 600;
        font-size: 0.9rem;
    }
    

    
    /* Footer */
    .footer {
        border-top: 1px solid rgba(8, 145, 178, 0.2);
        margin-top: 3rem;
        padding-top: 1.5rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Tabs - Clean Professional Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #6b7280;
        padding-bottom: 1rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb;
        border-bottom: 3px solid #2563eb;
    }
    
    /* FIX: Force Input Visibility */
    div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #0f172a !important;
    }
    
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD ASSETS
# =============================================
@st.cache_resource
def load_assets():
    """Load ML model and scaler with compatibility handling"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        
        # Check if model has required attributes, add them if missing (sklearn version compatibility)
        if not hasattr(model, 'multi_class'):
            model.multi_class = 'ovr'
        if not hasattr(model, 'n_iter_'):
            model.n_iter_ = np.array([100])
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_assets()

# =============================================
# HEADER
# =============================================
st.markdown("""
<div class="dashboard-header">
    <div>
        <h1 class="header-title">Heart Disease Prediction</h1>
        <p class="header-subtitle">Framingham Heart Study Predictive Model • Logistic Regression</p>
    </div>
    <div class="creator-tag">
        🏥 Project by Partha Sarathi R
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    # Display custom logo
    logo_path = "heart_disease_logo.png"
    try:
        st.image(logo_path, width=120)
    except:
        st.image("https://img.icons8.com/fluency/96/000000/cardiology.png", width=80)
    
    st.markdown("### 🏥 Clinical Dashboard")
    st.markdown("Enter patient vitals and demographics to generate a comprehensive 10-year CHD risk profile using advanced ML algorithms.")
    
    st.markdown("---")
    st.markdown("##### ⚙️ Model Specifications")
    st.caption("**Algorithm:** Logistic Regression")
    st.caption("**Training Accuracy:** 86.60%")
    st.caption("**Validation Set:** 1,060 samples")
    st.caption("**Dataset:** Framingham Heart Study")
    
    st.markdown("---")
    st.markdown("##### 👨‍💻 Developer Information")
    st.caption("**Developer:** Partha Sarathi R")
    st.caption("**Project:** Heart Disease Prediction System")
    st.caption("**Year:** 2026")
    st.caption("**Version:** 2.0.0")

# =============================================
# MAIN INTERFACE
# =============================================

if model:
    # Use Tabs for Organization
    tab_predict, tab_stats, tab_info = st.tabs(["Patient Evaluation", "Model Performance", "Study Details"])
    
    with tab_predict:
        # Input Section
        with st.container():
            col_left, col_mid, col_right = st.columns(3)
            
            # --- COLUMN 1: DEMOGRAPHICS & LIFESTYLE ---
            with col_left:
                st.markdown('<div class="input-card"><div class="section-title">01. Patient Profile</div>', unsafe_allow_html=True)
                
                gender = st.selectbox("Gender", ["Male", "Female"])
                male = 1 if gender == "Male" else 0
                
                age = st.number_input("Age", 30, 80, 50, help="Patient age in years")
                
                smoker_status = st.radio("Smoking History", ["Non-Smoker", "Current Smoker"], horizontal=True)
                currentSmoker = 1 if smoker_status == "Current Smoker" else 0
                
                if currentSmoker:
                    cigsPerDay = st.slider("Cigarettes / Day", 1, 70, 10)
                else:
                    cigsPerDay = 0
                
                st.markdown('</div>', unsafe_allow_html=True)

            # --- COLUMN 2: CLINICAL VITALS ---
            with col_mid:
                st.markdown('<div class="input-card"><div class="section-title">02. Clinical Vitals</div>', unsafe_allow_html=True)
                
                sysBP = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
                diaBP = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
                
                bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 25.0, format="%.1f")
                heartRate = st.slider("Resting Heart Rate (bpm)", 40, 120, 72)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            # --- COLUMN 3: LAB RESULTS ---
            with col_right:
                st.markdown('<div class="input-card"><div class="section-title">03. Lab Results & History</div>', unsafe_allow_html=True)
                
                totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
                glucose = st.number_input("Glucose (mg/dL)", 40, 400, 85)
                
                st.markdown('<div class="section-title" style="margin-top:1rem;">04. Medical History</div>', unsafe_allow_html=True)
                
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    bp_meds = st.checkbox("BP Meds")
                    diabetes_hist = st.checkbox("Diabetes")
                with h_col2:
                    stroke_hist = st.checkbox("Prior Stroke")
                    hyp_hist = st.checkbox("Hypertension")
                
                # Convert booleans to int
                BPMeds = 1 if bp_meds else 0
                diabetes = 1 if diabetes_hist else 0
                prevalentStroke = 1 if stroke_hist else 0
                prevalentHyp = 1 if hyp_hist else 0
                
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # PREDICTION ACTION
        col_space1, col_action, col_space2 = st.columns([1, 2, 1])
        with col_action:
            predict_btn = st.button("Generate Risk Assessment Report")

        # LOGIC
        if predict_btn:
            try:
                input_df = pd.DataFrame({
                    'male': [male], 'age': [age], 'currentSmoker': [currentSmoker], 
                    'cigsPerDay': [cigsPerDay], 'BPMeds': [BPMeds], 'prevalentStroke': [prevalentStroke],
                    'prevalentHyp': [prevalentHyp], 'diabetes': [diabetes], 'totChol': [totChol],
                    'sysBP': [sysBP], 'diaBP': [diaBP], 'BMI': [bmi], 'heartRate': [heartRate],
                    'glucose': [glucose]
                })
                
                scaled_input = scaler.transform(input_df)
                prob = model.predict_proba(scaled_input)[0][1] * 100
            except AttributeError as e:
                st.error(f"⚠️ Model compatibility error: {str(e)}. Please retrain the model with the current sklearn version.")
                st.stop()
            except Exception as e:
                st.error(f"⚠️ Prediction error: {str(e)}")
                st.stop()
            
            # --- RESULTS SECTION ---
            st.markdown("---")
            st.markdown("### Assessment Report")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                # Determine Styling with New Color Palette
                if prob < 20:
                    status_color = "#059669" # Emerald Green
                    status_text = "LOW RISK"
                    rec_text = "✅ Patient falls within the low-risk category. Continue maintaining a healthy lifestyle with regular check-ups."
                elif prob < 50:
                    status_color = "#d97706" # Amber
                    status_text = "ELEVATED RISK"
                    rec_text = "⚠️ Moderate risk factors identified. Lifestyle modifications and regular monitoring recommended."
                else:
                    status_color = "#e11d48" # Rose Red
                    status_text = "HIGH RISK"
                    rec_text = "🚨 Significant risk factors present. Immediate clinical intervention and comprehensive treatment plan may be required."
                    
                st.markdown(f"""
                <div class="report-container" style="border-top-color: {status_color};">
                    <p class="risk-label">10-Year CHD Probability</p>
                    <p style="color: {status_color}; font-size: 3.5rem; font-weight: 700; margin: 0;">{prob:.1f}%</p>
                    <p style="color: {status_color}; font-weight: 600; font-size: 1.2rem; margin-top: 0;">{status_text}</p>
                    <p style="color: #64748b; margin-top: 1rem; font-size: 0.95rem;">{rec_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with r_col2:
                # Enhanced Professional Gauge with Gradient
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={
                        'font': {'color': status_color, 'size': 48, 'family': 'Inter'},
                        'suffix': '%'
                    },
                    gauge={
                        'axis': {
                            'range': [0, 100], 
                            'tickwidth': 2, 
                            'tickcolor': "#0891b2", 
                            'tickfont': {'color': "#0f172a", 'size': 12, 'family': 'Inter'}
                        },
                        'bar': {'color': status_color, 'thickness': 0.8},
                        'bgcolor': "rgba(255, 255, 255, 0.3)",
                        'borderwidth': 3,
                        'bordercolor': "#0891b2",
                        'steps': [
                            {'range': [0, 20], 'color': "#d1fae5"},    # Lighter Green
                            {'range': [20, 50], 'color': "#fef3c7"},   # Lighter Amber
                            {'range': [50, 100], 'color': "#fecdd3"}   # Lighter Rose
                        ],
                        'threshold': {
                            'line': {'color': status_color, 'width': 4},
                            'thickness': 0.85,
                            'value': prob
                        }
                    }
                ))
                fig.update_layout(
                    height=320, 
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Inter", 'color': "#0f172a"}
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_stats:
        st.markdown("### 📊 Model Performance Metrics")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Overall Accuracy", "86.60%", "+1.2%")
        m_col2.metric("ROC-AUC Score", "0.728", "-0.01")
        m_col3.metric("Dataset Size", "4,238", "Patients")
        
        st.markdown("### 🔍 Feature Importance Analysis")
        # Enhanced feature data
        feature_importance = pd.DataFrame({
            'Factor': ['Age', 'Cigarettes/Day', 'Systolic BP', 'Gender (Male)', 'Glucose'],
            'Coefficient': [0.545, 0.290, 0.290, 0.220, 0.188]
        })
        
        fig_bar = px.bar(feature_importance, x='Coefficient', y='Factor', orientation='h',
                        title="Top 5 Predictive Risk Factors", text_auto='.3f')
        fig_bar.update_layout(
            plot_bgcolor="rgba(255, 255, 255, 0.5)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font={'color': '#0f172a', 'family': 'Inter', 'size': 13},
            title_font={'size': 18, 'color': '#0891b2', 'family': 'Inter'},
            xaxis=dict(showgrid=True, gridcolor='rgba(8, 145, 178, 0.1)', title="Coefficient Value"),
            yaxis=dict(title="Risk Factor"),
            height=350,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_bar.update_traces(
            marker=dict(
                color=['#0891b2', '#06b6d4', '#22d3ee', '#67e8f9', '#a5f3fc'],
                line=dict(color='#0e7490', width=2)
            ),
            textfont=dict(size=12, color='white', family='Inter')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_info:
        st.markdown("### 📚 About the Framingham Heart Study")
        st.write("""
        The **Framingham Heart Study** is a landmark, long-term cardiovascular cohort study that began in 1948 in Framingham, Massachusetts. 
        With over 5,200 original participants and now spanning three generations, it has been instrumental in identifying major cardiovascular 
        risk factors and advancing preventive medicine.
        
        This prediction system leverages decades of research data to provide evidence-based risk assessments for coronary heart disease.
        """)
        
        st.info("🎓 **Academic Project 2026** • Developed by **Partha Sarathi R** for Advanced Data Science and Machine Learning coursework.")

else:
    st.error("Model file/scaler not found. Please verify the file paths.")

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer">
    © 2026 Partha Sarathi R • Heart Disease Prediction System • Framingham Model • v2.0.0
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
