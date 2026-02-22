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

import numpy as np
import joblib

# ───────────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────────────────────
# Load Model
# ───────────────────────────────────────────────
model = joblib.load("model.pkl")

# ───────────────────────────────────────────────
# Global CSS — Dark Glassmorphism Theme
# ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-dark:      #050d1a;
    --bg-card:      rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.10);
    --accent:       #3b82f6;
    --accent2:      #8b5cf6;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --text-primary: #f0f4ff;
    --text-muted:   #8b96b0;
}

/* ── Full-page background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top left, #0a1628 0%, #050d1a 55%, #0c0820 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
section.main > div { padding-top: 1.5rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

/* ── Remove default padding on block col ── */
[data-testid="column"] { padding: 0.4rem !important; }

/* ── Streamlit label text ── */
label, .stSlider label, .stSelectbox label {
    color: var(--text-muted) !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 0.75rem !important;
}
input[type="number"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.20) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .st-emotion-cache-1e3f0d7 { color: var(--accent) !important; }
[data-testid="stSlider"] div[role="slider"] { background: var(--accent) !important; }

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 35px rgba(59,130,246,0.50) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

/* ── Metric widget ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(12px);
}
[data-testid="stMetricLabel"] p { color: var(--text-muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 2.4rem !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Helper: Glass Card wrapper
# ───────────────────────────────────────────────
def glass_card(html_content: str, extra_style: str = ""):
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 1rem;
        {extra_style}
    ">
        {html_content}
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem;">
    <div style="display:inline-flex; align-items:center; gap:14px; margin-bottom:12px;">
        <span style="font-size:3rem; line-height:1;">🫀</span>
        <div style="text-align:left;">
            <h1 style="
                margin:0;
                font-family:'Inter',sans-serif;
                font-size:2.4rem;
                font-weight:800;
                background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing:-0.02em;
                line-height:1.1;
            ">AI Health Risk Predictor</h1>
            <p style="margin:4px 0 0; color:#8b96b0; font-size:0.92rem; font-weight:400; letter-spacing:0.01em;">
                Powered by Lasso &amp; Linear Regression
            </p>
        </div>
    </div>
    <p style="
        color:#8b96b0;
        font-size:0.95rem;
        max-width:600px;
        margin: 0 auto;
        line-height:1.65;
    ">
        Enter clinical indicators below to predict a patient's overall health risk score using a trained machine-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# ── Section: Clinical Indicators ──
# ───────────────────────────────────────────────
glass_card("""
<h3 style="margin:0 0 1.2rem; font-family:'Inter',sans-serif; font-size:1.05rem; font-weight:700;
           color:#60a5fa; letter-spacing:0.04em; text-transform:uppercase; display:flex; align-items:center; gap:8px;">
    <span>🩺</span> Clinical Indicators
</h3>
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p style='color:#8b96b0;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;'>Patient Demographics</p>", unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=120, step=1)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72, step=1)

with col2:
    st.markdown("<p style='color:#8b96b0;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;'>Lab Results</p>", unsafe_allow_html=True)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0.0, max_value=500.0, value=80.0, step=0.5, format="%.1f")

with col3:
    st.markdown("<p style='color:#8b96b0;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;'>Lifestyle Factors</p>", unsafe_allow_html=True)
    activity_level = st.slider("Activity Level (1–10)", min_value=1, max_value=10, value=5)
    diet_quality   = st.slider("Diet Quality (1–10)",   min_value=1, max_value=10, value=5)
    smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
    alcohol_intake = st.number_input("Alcohol Intake (units/week)", min_value=0.0, max_value=50.0, value=2.0, step=0.5, format="%.1f")

# Convert smoking
smoking_enc = 1 if smoking_status == "Yes" else 0

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Predict Button
# ───────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔍  Analyze Health Risk")

# ───────────────────────────────────────────────
# Result Section
# ───────────────────────────────────────────────
if predict_clicked:
    input_data = np.array([[age, bmi, blood_pressure, cholesterol,
                            glucose, insulin, heart_rate,
                            activity_level, diet_quality,
                            smoking_enc, alcohol_intake]])
    prediction = model.predict(input_data)[0]

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Risk category config ──
    if prediction < 33:
        risk_label   = "Low Risk"
        risk_emoji   = "🟢"
        risk_color   = "#10b981"
        risk_bg      = "rgba(16,185,129,0.10)"
        risk_border  = "rgba(16,185,129,0.35)"
        risk_advice  = "Patient shows healthy indicators. Keep up the great lifestyle!"
        progress_pct = max(int(prediction / 33 * 100), 4)
        bar_color    = "#10b981"
    elif prediction < 66:
        risk_label   = "Moderate Risk"
        risk_emoji   = "🟡"
        risk_color   = "#f59e0b"
        risk_bg      = "rgba(245,158,11,0.10)"
        risk_border  = "rgba(245,158,11,0.35)"
        risk_advice  = "Some risk factors detected. Lifestyle improvement is recommended."
        progress_pct = max(int((prediction - 33) / 33 * 100), 4)
        bar_color    = "#f59e0b"
    else:
        risk_label   = "High Risk"
        risk_emoji   = "🔴"
        risk_color   = "#ef4444"
        risk_bg      = "rgba(239,68,68,0.10)"
        risk_border  = "rgba(239,68,68,0.35)"
        risk_advice  = "Significant risk indicators found. Medical consultation strongly advised."
        progress_pct = max(int((prediction - 66) / 34 * 100), 4)
        bar_color    = "#ef4444"

    # ── Score Card ──
    st.markdown(f"""
    <div style="
        background: {risk_bg};
        border: 1px solid {risk_border};
        border-radius: 20px;
        padding: 2rem 2.2rem;
        backdrop-filter: blur(16px);
        text-align: center;
        margin-bottom: 1.2rem;
    ">
        <p style="margin:0 0 6px; color:#8b96b0; font-size:0.78rem; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.08em;">Predicted Health Risk Score</p>
        <h2 style="margin:0; font-family:'Inter',sans-serif; font-size:4rem; font-weight:900;
                   color:{risk_color}; line-height:1; letter-spacing:-0.03em;">{prediction:.1f}</h2>
        <div style="margin: 1rem auto 0.6rem; max-width:340px;
                    height:8px; background:rgba(255,255,255,0.08); border-radius:99px; overflow:hidden;">
            <div style="height:100%; width:{progress_pct}%;
                        background: linear-gradient(90deg, {bar_color}99, {bar_color});
                        border-radius:99px; transition:width 0.6s ease;"></div>
        </div>
        <p style="margin:6px 0 0; font-size:1.15rem; font-weight:700; color:{risk_color};">
            {risk_emoji} {risk_label}
        </p>
        <p style="margin:6px 0 0; color:#8b96b0; font-size:0.9rem;">{risk_advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick Summary Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Age", f"{age} yrs")
    with m2:
        st.metric("BMI", f"{bmi:.1f}")
    with m3:
        st.metric("Glucose", f"{glucose} mg/dL")
    with m4:
        st.metric("Cholesterol", f"{cholesterol} mg/dL")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <p style="text-align:center; color:#4b5563; font-size:0.78rem; margin-top:1rem;">
        ⚠️ <strong style='color:#6b7280'>Disclaimer:</strong>
        This tool is intended for educational and research purposes only and does not constitute a medical diagnosis.
        Always consult a qualified healthcare professional.
    </p>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="
    text-align:center;
    padding: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    color:#3d4d66;
    font-size:0.78rem;
    letter-spacing:0.03em;
">
    Built with ❤️ using <strong style='color:#4b6a8f'>Streamlit</strong> &nbsp;|&nbsp;
    Lasso &amp; Linear Regression Model &nbsp;|&nbsp;
    <strong style='color:#4b6a8f'>Partha Sarathi R</strong>
</div>
""", unsafe_allow_html=True)\n\nst.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)\n