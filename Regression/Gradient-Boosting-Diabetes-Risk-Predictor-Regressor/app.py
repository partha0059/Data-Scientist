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

import joblib
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS Styling ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Gradient */
    .stApp {
        background: #000000;
        background-attachment: fixed;
    }
    
    /* Animated Background Pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.4) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.3) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: backgroundShift 15s ease-in-out infinite;
    }
    
    @keyframes backgroundShift {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        animation: fadeInDown 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    h2 {
        font-family: 'Poppins', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #c7d2fe !important;
        font-weight: 600 !important;
    }
    
    /* Subtitle Text */
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 3rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Glassmorphic Card Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(0, 0, 0, 0.2),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.2);
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Input Fields Styling */
    .stNumberInput > div > div > input {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus {
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        padding: 1.2rem 5rem !important;
        border-radius: 15px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 2rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Success/Result Box */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
        animation: resultPulse 0.6s ease-out;
    }
    
    @keyframes resultPulse {
        0% {
            transform: scale(0.95);
            opacity: 0;
        }
        50% {
            transform: scale(1.02);
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .stSuccess p {
        color: #d1fae5 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar Text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: #e0e7ff !important;
    }
    
    /* Info Box in Sidebar */
    .sidebar-info {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8b5cf6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(139, 92, 246, 0.2);
        color: #a5b4fc;
        font-size: 0.95rem;
    }
    
    .footer-gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Info/Warning Messages */
    .stAlert {
        background: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #e0e7ff !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Column Gap */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.markdown("""
    <div class="sidebar-info">
        <p>This application uses <strong>Gradient Boosting Regression</strong> to predict diabetes outcome values based on patient data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Enter patient data in the input fields
    2. All values should be numerical
    3. Click the **Predict** button
    4. View the predicted outcome value
    """)
    
    st.markdown("### 🔬 Model Information")
    st.markdown("""
    - **Algorithm**: Gradient Boosting Regressor
    - **Framework**: Scikit-learn
    - **Purpose**: Medical prediction
    """)
    
    st.markdown("---")
    st.markdown("""
    <p style='color: #e0e7ff; font-size: 0.9rem; margin: 0.5rem 0; line-height: 1.6;'>
        <strong>👨💻 Developer Information</strong><br>
        Developer: Partha Sarathi R<br>
        Project: Diabetes Risk Predictor<br>
        Email: ayyapparaja227@gmail.com
    </p>
    """, unsafe_allow_html=True)

# ---------------- Main Header ----------------
st.markdown("<h1>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Machine Learning for Medical Diagnostics</p>", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_gradient_boosting_model.pkl")
    feature_columns = joblib.load("diabetes_feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ---------------- User Input Section ----------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("## 🔢 Patient Data Input")
st.markdown("<p style='color: #a5b4fc; margin-bottom: 2rem;'>Please enter the patient's medical parameters below</p>", unsafe_allow_html=True)

# Create responsive columns for inputs
num_features = len(feature_columns)
cols_per_row = 3
input_data = {}

# Display inputs in rows of 3 columns
for i in range(0, num_features, cols_per_row):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i + j
        if idx < num_features:
            feature = feature_columns[idx]
            with cols[j]:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.01,
                    format="%.2f"
                )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# ---------------- Prediction Button ----------------
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    predict_clicked = st.button("🔍 Analyze Health Metrics")

if predict_clicked:
    with st.spinner("🔄 Analyzing patient data..."):
        prediction = model.predict(input_df)[0]
        
        # Determine risk level and styling
        if prediction < 100:
            status_text = "Low"
            status_color = "#10b981"  # Green
            message = "Your diabetes risk is low. Maintain healthy lifestyle habits."
        elif prediction < 200:
            status_text = "Moderate"
            status_color = "#f59e0b"  # Orange
            message = "Your diabetes risk is moderate. Consider lifestyle improvements."
        else:
            status_text = "High"
            status_color = "#ec4899"  # Pink/Coral
            message = "Your diabetes risk is elevated. Please consult a healthcare provider."
        
        st.markdown(f"""
        <div style='
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            animation: resultPulse 0.6s ease-out;
        '>
            <h2 style='color: #ffffff; margin-bottom: 1.5rem; font-size: 1.8rem; font-weight: 700;'>
                🎯 Risk Assessment Complete
            </h2>
            <p style='font-size: 4rem; font-weight: 700; color: #ffffff; margin: 1rem 0; line-height: 1;'>
                {prediction:.2f}
            </p>
            <p style='color: {status_color}; font-size: 1.5rem; margin: 1rem 0; font-weight: 600;'>
                Risk Level: {status_text}
            </p>
            <p style='color: rgba(255, 255, 255, 0.9); font-size: 1rem; margin-top: 1.5rem; opacity: 0.95;'>
                {message}
            </p>
            <div style='
                margin-top: 2rem;
                padding-top: 1.5rem;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            '>
                <p style='color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; line-height: 1.5;'>
                    <em>Note: This is an ML estimation based on learned patterns from medical data. 
                    Typical prediction accuracy: ±16 points. Always consult healthcare professionals 
                    for medical decisions.</em>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p style='margin-bottom: 0.5rem;'>
        <span style='font-size: 1.1rem;'>🎓</span> 
        <span style='color: #ffffff; font-weight: 700; font-size: 1.1rem;'>Created by Partha Sarathi R</span>
    </p>
    <p style='color: #f0f0f0; font-size: 0.9rem; font-weight: 500;'>
        <strong>Gradient Boosting Regressor</strong> | Built with ❤️ using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
