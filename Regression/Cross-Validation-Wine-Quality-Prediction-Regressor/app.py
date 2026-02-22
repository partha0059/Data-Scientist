import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
\nimport streamlit as st

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
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="K-Fold Cross-Validation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark Glassmorphism theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - white */
    .stApp {
        background: #ffffff;
        background-attachment: fixed;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glass card styling */
    .glass-card {
        background: rgba(138, 43, 226, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(138, 43, 226, 0.15);
        padding: 30px;
        box-shadow: 0 4px 12px 0 rgba(138, 43, 226, 0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px 0 rgba(138, 43, 226, 0.2);
        border: 1px solid rgba(138, 43, 226, 0.3);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.15) 0%, rgba(75, 0, 130, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(138, 43, 226, 0.2);
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 40px 0 rgba(138, 43, 226, 0.4);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #b19cd9 0%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #333333;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Title styling */
    .main-title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #000000;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #555555;
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8a2be2 0%, #9370db 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #9370db 0%, #8a2be2 100%);
        box-shadow: 0 6px 20px rgba(138, 43, 226, 0.5);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(248, 248, 250, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(138, 43, 226, 0.15);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(138, 43, 226, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(0, 0, 0, 0.6);
        font-weight: 500;
        border: none;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8a2be2 0%, #9370db 100%);
        color: white;
    }
    
    /* Input styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #8a2be2 0%, #9370db 100%);
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: white;
        border: 1px solid rgba(138, 43, 226, 0.3);
        border-radius: 8px;
        color: #333;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
    }
    
    p, span, label, div {
        color: #333333;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(65, 105, 225, 0.08) 0%, rgba(30, 144, 255, 0.08) 100%);
        border-left: 4px solid #4169e1;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #2c2c2c;
    }
    
    .info-box ul {
        color: #2c2c2c;
        margin: 10px 0;
        padding-left: 20px;
    }
    
    .info-box li {
        color: #2c2c2c;
        margin: 5px 0;
        line-height: 1.6;
    }
    
    .info-box b {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, rgba(34, 139, 34, 0.08) 0%, rgba(50, 205, 50, 0.08) 100%);
        border-left: 4px solid #32cd32;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #2c2c2c;
    }
    
    .success-box b {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(0, 0, 0, 0.5);
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid rgba(138, 43, 226, 0.15);
    }
    
    /* Project info box */
    .project-info {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.08) 0%, rgba(147, 112, 219, 0.08) 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(138, 43, 226, 0.2);
    }
    
    .project-info h4 {
        color: #8a2be2;
        margin-bottom: 10px;
        font-size: 16px;
    }
    
    .project-info p {
        color: #444444;
        font-size: 13px;
        line-height: 1.6;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load Wine Quality dataset"""
    try:
        data = pd.read_csv('winequality-red.csv', sep=';')
    except FileNotFoundError:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url, sep=';')
    return data

@st.cache_resource
def get_model():
    """Load or train the model"""
    if os.path.exists('wine_quality_model.pkl'):
        model = joblib.load('wine_quality_model.pkl')
    else:
        # Train model if not exists
        data = load_data()
        X = data.drop('quality', axis=1)
        y = data['quality']
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, 'wine_quality_model.pkl')
    return model

def perform_cross_validation(X, y, n_splits, random_state):
    """Perform K-Fold Cross-Validation"""
    model = LinearRegression()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    rmse_scores = np.sqrt(-cv_scores)
    
    return rmse_scores

def create_rmse_bar_chart(rmse_scores, n_splits):
    """Create bar chart for RMSE scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(n_splits)],
        y=rmse_scores,
        marker=dict(
            color=rmse_scores,
            colorscale=[[0, '#8a2be2'], [0.5, '#9370db'], [1, '#b19cd9']],
            line=dict(color='rgba(138, 43, 226, 0.5)', width=2)
        ),
        text=[f'{score:.4f}' for score in rmse_scores],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=rmse_scores.mean(),
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.5)",
        annotation_text=f"Mean: {rmse_scores.mean():.4f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title={
            'text': f'RMSE per Fold ({n_splits}-Fold Cross-Validation)',
            'font': {'size': 20, 'color': '#1a1a1a', 'family': 'Inter'}
        },
        xaxis={
            'title': {'text': 'Fold', 'font': {'size': 14, 'color': '#1a1a1a'}},
            'gridcolor': 'rgba(138, 43, 226, 0.15)',
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        yaxis={
            'title': {'text': 'RMSE', 'font': {'size': 14, 'color': '#1a1a1a'}},
            'gridcolor': 'rgba(138, 43, 226, 0.15)',
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        plot_bgcolor='rgba(255, 255, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font={'color': '#1a1a1a', 'family': 'Inter', 'size': 12},
        height=400
    )
    
    return fig

def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart"""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    fig = go.Figure()
    
    colors = ['#8a2be2' if x > 0 else '#ff6b6b' for x in importance['Coefficient']]
    
    fig.add_trace(go.Bar(
        y=importance['Feature'],
        x=importance['Coefficient'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        text=[f'{val:.3f}' for val in importance['Coefficient']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Feature Coefficients',
            'font': {'size': 20, 'color': '#1a1a1a', 'family': 'Inter'}
        },
        xaxis={
            'title': {'text': 'Coefficient Value', 'font': {'size': 14, 'color': '#1a1a1a'}},
            'gridcolor': 'rgba(138, 43, 226, 0.15)',
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        yaxis={
            'gridcolor': 'rgba(138, 43, 226, 0.15)',
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        plot_bgcolor='rgba(255, 255, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font={'color': '#1a1a1a', 'family': 'Inter', 'size': 12},
        height=500
    )
    
    return fig

def create_distribution_chart(rmse_scores):
    """Create distribution chart for RMSE scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=rmse_scores,
        name='RMSE Distribution',
        marker=dict(color='#8a2be2'),
        boxmean='sd',
        hovertemplate='RMSE: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'RMSE Score Distribution',
            'font': {'size': 20, 'color': '#1a1a1a', 'family': 'Inter'}
        },
        xaxis={
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        yaxis={
            'title': {'text': 'RMSE', 'font': {'size': 14, 'color': '#1a1a1a'}},
            'gridcolor': 'rgba(138, 43, 226, 0.15)',
            'tickfont': {'size': 12, 'color': '#000000'}
        },
        plot_bgcolor='rgba(255, 255, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font={'color': '#1a1a1a', 'family': 'Inter', 'size': 12},
        showlegend=False,
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-title">📊 K-Fold Cross-Validation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Wine Quality Prediction with Linear Regression</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Number of folds input
        n_splits = st.number_input(
            "Number of Folds",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Enter the number of folds for K-Fold Cross-Validation (3-20)"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Set random state for reproducibility"
        )
        
        run_cv = st.button("🚀 Run Cross-Validation", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📚 Project Information")
        
        st.markdown("**🎯 Overview**")
        st.write("K-Fold Cross-Validation demonstration for evaluating machine learning models.")
        
        st.markdown("**📊 Dataset**")
        st.write("UCI Wine Quality with 1,599 samples and 11 features.")
        
        st.markdown("**🤖 Algorithm**")
        st.write("Linear Regression with K-Fold validation.")
        
        st.markdown("**💡 Features**")
        st.write("• Adjustable folds (3-20)")
        st.write("• Real-time validation")
        st.write("• Interactive charts")
        st.write("• Feature analysis")
    
    # Load data
    data = load_data()
    X = data.drop('quality', axis=1)
    y = data['quality']
    model = get_model()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Quality", "📈 Cross-Validation", "📊 Model Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 🍷 Predict Wine Quality")
        st.markdown("Enter the physicochemical properties of the wine to predict its quality rating.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9, step=0.1)
            chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076, step=0.001)
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
        
        with col2:
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0, step=1.0)
            density = st.number_input("Density", min_value=0.98, max_value=1.01, value=0.9978, step=0.0001, format="%.4f")
            ph = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.51, step=0.01)
            sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56, step=0.01)
            alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=9.4, step=0.1)
        
        if st.button("🎯 Predict Wine Quality", use_container_width=True):
            # Create input array
            input_data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, ph, sulphates, alcohol
            ]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Determine quality category
            if prediction < 5:
                quality_text = "Low Quality"
                quality_color = "#ff6b6b"
            elif prediction < 6:
                quality_text = "Medium Quality"
                quality_color = "#ffd93d"
            elif prediction < 7:
                quality_text = "Good Quality"
                quality_color = "#6bcf7f"
            else:
                quality_text = "Excellent Quality"
                quality_color = "#8a2be2"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quality_color}20 0%, {quality_color}10 100%);
                border-left: 4px solid {quality_color};
                border-radius: 12px;
                padding: 25px;
                text-align: center;
            ">
                <h2 style="color: {quality_color}; margin: 0 0 10px 0;">Predicted Quality Score</h2>
                <h1 style="color: {quality_color}; font-size: 64px; margin: 10px 0;">{prediction:.2f}</h1>
                <h3 style="color: #333; margin: 10px 0 0 0;">{quality_text}</h3>
                <p style="color: #666; margin-top: 15px; font-size: 14px;">
                    Quality scale ranges from 0 (worst) to 10 (best)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show input summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Input Summary")
            input_df = pd.DataFrame({
                'Feature': X.columns.tolist(),
                'Your Input': input_data[0],
                'Dataset Average': X.mean().values,
                'Difference': input_data[0] - X.mean().values
            })
            st.dataframe(
                input_df.style.format({
                    'Your Input': '{:.4f}',
                    'Dataset Average': '{:.4f}',
                    'Difference': '{:+.4f}'
                }),
                use_container_width=True
            )
    
    with tab2:
        if run_cv or 'rmse_scores' not in st.session_state:
            with st.spinner('Performing cross-validation...'):
                rmse_scores = perform_cross_validation(X, y, n_splits, int(random_state))
                st.session_state.rmse_scores = rmse_scores
                st.session_state.n_splits = n_splits
        
        if 'rmse_scores' in st.session_state:
            rmse_scores = st.session_state.rmse_scores
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Folds</div>
                    <div class="metric-value">{st.session_state.n_splits}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean RMSE</div>
                    <div class="metric-value">{rmse_scores.mean():.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{rmse_scores.std():.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Fold</div>
                    <div class="metric-value">{rmse_scores.argmin() + 1}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_rmse_bar_chart(rmse_scores, st.session_state.n_splits),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_distribution_chart(rmse_scores),
                    use_container_width=True
                )
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            results_df = pd.DataFrame({
                'Fold': [f'Fold {i+1}' for i in range(st.session_state.n_splits)],
                'RMSE': rmse_scores,
                'Difference from Mean': rmse_scores - rmse_scores.mean()
            })
            st.dataframe(
                results_df.style.format({'RMSE': '{:.4f}', 'Difference from Mean': '{:+.4f}'}),
                use_container_width=True
            )
            
            st.markdown(f"""
            <div class="success-box">
                <b>✓ Cross-Validation Complete</b><br>
                The model achieved an average RMSE of <b>{rmse_scores.mean():.4f}</b> 
                with a standard deviation of <b>{rmse_scores.std():.4f}</b> across {st.session_state.n_splits} folds.
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🎯 Feature Coefficients")
        st.plotly_chart(
            create_feature_importance_chart(model, X.columns.tolist()),
            use_container_width=True
        )
        
        # Dataset info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Dataset Information")
            st.markdown(f"""
            <div class="info-box">
                <b>Total Samples:</b> {len(data)}<br>
                <b>Features:</b> {len(X.columns)}<br>
                <b>Target Variable:</b> quality<br>
                <b>Quality Range:</b> {y.min()} - {y.max()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔧 Model Parameters")
            st.markdown(f"""
            <div class="info-box">
                <b>Algorithm:</b> Linear Regression<br>
                <b>Intercept:</b> {model.intercept_:.4f}<br>
                <b>Coefficients:</b> {len(model.coef_)}<br>
                <b>Fit Intercept:</b> True
            </div>
            """, unsafe_allow_html=True)
        
        # Feature statistics
        st.markdown("### 📈 Feature Statistics")
        st.dataframe(
            X.describe().style.format('{:.2f}'),
            use_container_width=True
        )
    
    with tab4:
        st.markdown("### 🎓 What is K-Fold Cross-Validation?")
        st.markdown("""
        <div class="info-box">
            K-Fold Cross-Validation is a robust model evaluation technique that:
            <ul>
                <li>Divides the dataset into K equal-sized folds</li>
                <li>Trains the model K times, each time using K-1 folds for training</li>
                <li>Uses the remaining fold for validation</li>
                <li>Averages the results to get a more reliable performance estimate</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 About This Application")
        st.markdown("""
        <div class="info-box">
            This application demonstrates K-Fold Cross-Validation using the UCI Wine Quality dataset.
            The Linear Regression model predicts wine quality based on physicochemical properties.
            
            <br><br>
            <b>Dataset Features:</b>
            <ul>
                <li>Fixed Acidity</li>
                <li>Volatile Acidity</li>
                <li>Citric Acid</li>
                <li>Residual Sugar</li>
                <li>Chlorides</li>
                <li>Free Sulfur Dioxide</li>
                <li>Total Sulfur Dioxide</li>
                <li>Density</li>
                <li>pH</li>
                <li>Sulphates</li>
                <li>Alcohol</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        <div class="success-box">
            <b>1.</b> Adjust the number of folds using the slider in the sidebar<br>
            <b>2.</b> Set the random state for reproducibility<br>
            <b>3.</b> Click "Run Cross-Validation" to perform the analysis<br>
            <b>4.</b> View the results in the Cross-Validation tab<br>
            <b>5.</b> Explore feature coefficients in the Model Analysis tab
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Created by Partha Sarathi R | Wine Quality Dataset from UCI Machine Learning Repository
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
