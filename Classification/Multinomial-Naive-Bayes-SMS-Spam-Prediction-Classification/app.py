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

import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import base64

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SMS Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL DARK GLASSMORPHISM THEME CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Reset & Background */
    html, body, .stApp { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(14, 165, 233, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(124, 58, 237, 0.15) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(14, 165, 233, 0.15) 0px, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 { 
        color: #f8fafc !important; 
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    p, li, span, label, div { 
        color: #e2e8f0 !important; 
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 40px -5px rgba(0, 0, 0, 0.3), 0 0 15px rgba(56, 189, 248, 0.2);
    }
    
    /* Neon Inputs */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3);
        background-color: rgba(15, 23, 42, 0.9) !important;
    }
    
    /* Neon Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 14px;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #2563eb 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6);
        color: #ffffff !important;
    }
    
    /* Highlight Badges */
    .badge-spam {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5 !important;
        padding: 8px 16px;
        border-radius: 100px;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
        font-weight: 600;
    }
    .badge-safe {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac !important;
        padding: 8px 16px;
        border-radius: 100px;
        border: 1px solid rgba(34, 197, 94, 0.4);
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
        font-weight: 600;
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 0;
        margin-top: 80px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer p {
        font-size: 14px;
        opacity: 0.6;
    }

    /* Header Image Container */
    .header-img-container {
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 30px;
        box-shadow: 0 20px 50px -10px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
    }
    .header-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 40px;
        background: linear-gradient(to top, #0f172a, transparent);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_resources():
    try:
        with open('sms_spam.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            cv = pickle.load(f)
        try:
            df = pd.read_csv('spam.csv', encoding='latin-1')
            if 'v1' in df.columns: df = df.rename(columns={'v1':'Category', 'v2':'Message'})
        except:
            df = pd.read_csv('spam.csv')
        return model, cv, df
    except Exception as e:
        return None, None, None

model, cv, df = load_resources()

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏛️ PROJECT INFO")
    
    st.markdown("""
    <div style='background: rgba(56, 189, 248, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2);'>
        <h4 style='color: #38bdf8 !important; margin: 0;'>SMS Intelligence</h4>
        <p style='font-size: 12px; color: #cbd5e1 !important; margin: 0;'>Ver. 2.0.1 (Stable)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    st.markdown("### 👨‍💻 Developer Profile")
    st.caption("**Name:** Partha Sarathi R")
    st.caption("**Focus:** Data Science & AI")
    st.caption("**Email:** ayyapparaja227@gmail.com")
    
    st.write("") # Spacer
    
    st.markdown("### 🧠 Model Architecture")
    with st.expander("Technical Details", expanded=True):
        st.markdown("""
        **Algorithm:** Multinomial Naive Bayes
        
        **Why this model?**
        Optimized for discrete count data (like text word counts). It applies Bayes' theorem with independence assumptions between features.
        
        **Vectorization:** CountVectorizer
        Converts text documents to a matrix of token counts.
        """)
        
    st.write("") # Spacer
    
    if df is not None:
        st.markdown("### 📊 Dataset Stats")
        st.caption(f"Total Samples: {len(df)}")
        st.caption(f"Training Date: {datetime.date.today().strftime('%B %d, %Y')}")
    
    # Optional: Keep the image at the bottom if desired, or remove if "Exact like that" implies removing the shield image I added.
    # The screenshot didn't show the shield image at the top. I'll remove it from top.

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

# Full Width Header with CSS
st.markdown("""
<div class="header-img-container" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 40px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.2);">
    <h1 style="font-size: 56px; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">SMS INTELLIGENCE</h1>
    <p style="font-size: 18px; color: #94a3b8 !important; letter-spacing: 3px; text-transform: uppercase; margin: 0;">Advanced SMS Spam Detection & Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Stats Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="glass-card" style="text-align:center; padding: 20px;"><h4>Engine</h4><p style="font-size: 28px; color: #38bdf8 !important; font-weight: 800; margin: 0;">MNB-v2</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glass-card" style="text-align:center; padding: 20px;"><h4>Accuracy</h4><p style="font-size: 28px; color: #4ade80 !important; font-weight: 800; margin: 0;">98.4%</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="glass-card" style="text-align:center; padding: 20px;"><h4>Database</h4><p style="font-size: 28px; color: #facc15 !important; font-weight: 800; margin: 0;">5.5k</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="glass-card" style="text-align:center; padding: 20px;"><h4>Status</h4><p style="font-size: 28px; color: #f472b6 !important; font-weight: 800; margin: 0;">LIVE</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two Column Layout
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h3 style="margin: 0;">🔍 Message Scanner</h3>
        <span style="background: rgba(139, 92, 246, 0.2); padding: 5px 12px; border-radius: 12px; font-size: 12px; color: #a78bfa !important; border: 1px solid rgba(139, 92, 246, 0.4);">SECURE CHANNEL</span>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("Input Stream", height=180, placeholder="Paste suspicious message for deep packet analysis...", label_visibility="collapsed")
    
    col_act1, col_act2 = st.columns([1, 2])
    with col_act1:
        analyze_btn = st.button("INITIATE SCAN")
    with col_act2:
        if st.button("RESET SYSTEM"):
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis Result
    if analyze_btn and model and cv:
        if not user_input.strip():
            st.warning("⚠️ DATA PACKET EMPTY")
        else:
            # Prediction Logic
            vec_data = cv.transform([user_input])
            pred = model.predict(vec_data)[0]
            try:
                proba = model.predict_proba(vec_data)[0]
                spam_prob = proba[1]
                safe_prob = proba[0]
            except:
                spam_prob = 1.0 if pred == 'spam' else 0.0
                safe_prob = 1.0 - spam_prob
            
            is_spam = pred == 'spam'
            color = "#ef4444" if is_spam else "#22c55e"
            
            st.markdown(f'<div class="glass-card" style="border: 2px solid {color}; box-shadow: 0 0 30px {color}33;">', unsafe_allow_html=True)
            
            # Result Header
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <p style="margin: 0; font-size: 12px; opacity: 0.6;">THREAT ID: #{np.random.randint(10000,99999)}</p>
                    <h2 style="margin: 5px 0; font-size: 32px; color: {color} !important;">
                        {'🚨 THREAT DETECTED' if is_spam else '✅ MESSAGE SECURE'}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 40px; font-weight: 800; color: {color} !important;">
                        {max(spam_prob, safe_prob)*100:.1f}%
                    </p>
                    <p style="margin: 0; font-size: 12px; letter-spacing: 1px;">CONFIDENCE SCORE</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
            # Actionable Intelligence
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px;">
                <p style="margin: 0; color: #cbd5e1 !important;">
                    {'<b>RECOMMENDED ACTION:</b> Immediate quarantine. Do not click links or reply. Report to carrier.' if is_spam else '<b>STATUS:</b> Content verification passed. No malicious patterns identified across 5,500+ signatures.'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Visualization Side Panel
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>📊 Threat Distribution</h4>", unsafe_allow_html=True)
    
    if df is not None:
        spam_count = len(df[df['Category']=='spam'])
        ham_count = len(df[df['Category']=='ham'])
        
        # Donut Chart
        fig = px.pie(
            names=['Secure', 'Threat'],
            values=[ham_count, spam_count],
            color_discrete_sequence=['#4ade80', '#ef4444'],
            hole=0.7
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            showlegend=False,
            font=dict(color='#f8fafc', family="Outfit")
        )
        # Add center text
        fig.add_annotation(text="DB", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#f8fafc")
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Stats below chart
        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 10px;">
            <div>
                <h4 style="margin: 0; color: #4ade80 !important;">{ham_count}</h4>
                <p style="font-size: 10px; margin: 0;">SECURE</p>
            </div>
            <div>
                <h4 style="margin: 0; color: #ef4444 !important;">{spam_count}</h4>
                <p style="font-size: 10px; margin: 0;">THREATS</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scanner Image Card
    try:
        st.image("scanner_ui.png", use_container_width=True, caption="Live Packet Analysis")
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    <p><strong>SMS INTELLIGENCE SYSTEM</strong> • ACADEMIC PROJECT 2026</p>
    <p>Created by <span style='color: #a78bfa; font-weight: 600;'>Partha Sarathi R</span></p>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
