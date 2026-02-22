"""
Employee Performance Rating Prediction App
A professional Streamlit application for predicting employee performance ratings
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
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Performance Rating Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark glassmorphism theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8b2d1 !important;
        font-weight: 600 !important;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .excellent-performance {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 5px solid #10b981;
    }
    
    .needs-improvement {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%);
        border-left: 5px solid #fb923c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a8b2d1;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #a8b2d1;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #93c5fd !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_models():
    """Load saved model and label encoder, or train one if not found."""
    try:
        model = joblib.load("random_forest_model.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except FileNotFoundError:
        # If files are missing (e.g., on Streamlit Cloud), run the training script
        try:
            from train_model import train_and_save_model
            model, le = train_and_save_model()
            return model, le
        except Exception as e:
            st.error(f"⚠️ Error training model: {e}")
            st.stop()

def predict_performance(model, le, input_data):
    """Make prediction using the trained model"""
    # Encode department
    dept_encoded = le.transform([input_data['Department']])[0]
    
    # Create feature array
    features = np.array([[
        input_data['Age'],
        input_data['Experience_Years'],
        dept_encoded,
        input_data['Salary'],
        input_data['Work_Hours'],
        input_data['Projects_Handled'],
        input_data['Training_Hours']
    ]])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability

def create_gauge_chart(confidence):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24, 'color': '#ffffff'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#00d4ff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a8b2d1"},
            'bar': {'color': "#00d4ff"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "#ffffff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(251, 146, 60, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(250, 204, 21, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff"},
        height=300
    )
    
    return fig

def create_feature_importance_chart(model):
    """Create feature importance bar chart"""
    feature_names = ['Age', 'Experience', 'Department', 'Salary', 'Work Hours', 'Projects', 'Training Hours']
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        title_font_size=20,
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>📊 Employee Performance Rating Predictor</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a8b2d1; font-size: 1.2rem; margin-top: 0;'>AI-Powered Performance Analysis with Random Forest</p>", 
                unsafe_allow_html=True)
    
    # Load models
    model, le = load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About This App")
        st.markdown("""
        This application uses a **Random Forest Classifier** to predict employee performance ratings based on:
        
        - 📅 Age & Experience
        - 🏢 Department
        - 💰 Salary
        - ⏰ Work Hours
        - 📋 Projects Handled
        - 📚 Training Hours
        """)
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Enter employee details
        2. Click **Predict Performance**
        3. View results and insights
        """)
        
        st.markdown("---")
        st.markdown("**Created by:** Partha Sarathi R")
        st.markdown("**Project:** Employee Performance Rating Predictor")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Performance", "📊 Model Analytics", "ℹ️ Information"])
    
    with tab1:
        st.markdown("### Employee Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("👤 Age", min_value=18, max_value=70, value=30, step=1,
                                 help="Employee's age in years")
            
            experience = st.number_input("💼 Experience (Years)", min_value=0, max_value=50, value=5, step=1,
                                        help="Total years of professional experience")
            
            department = st.selectbox("🏢 Department", 
                                     options=le.classes_,
                                     help="Employee's department")
            
            salary = st.number_input("💰 Salary (₹/Year)", min_value=20000, max_value=200000, 
                                    value=50000, step=5000,
                                    help="Annual salary in rupees")
        
        with col2:
            work_hours = st.number_input("⏰ Work Hours (per week)", min_value=20, max_value=60, 
                                        value=40, step=1,
                                        help="Average weekly work hours")
            
            projects = st.number_input("📋 Projects Handled", min_value=0, max_value=20, 
                                      value=3, step=1,
                                      help="Number of projects currently handling")
            
            training_hours = st.number_input("📚 Training Hours (per month)", min_value=0, max_value=100, 
                                           value=20, step=5,
                                           help="Monthly training hours")
        
        st.markdown("---")
        
        # Predict button
        if st.button("🚀 Predict Performance Rating", use_container_width=True):
            input_data = {
                'Age': age,
                'Experience_Years': experience,
                'Department': department,
                'Salary': salary,
                'Work_Hours': work_hours,
                'Projects_Handled': projects,
                'Training_Hours': training_hours
            }
            
            prediction, probability = predict_performance(model, le, input_data)
            
            # Results section
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                st.plotly_chart(create_gauge_chart(max(probability)), use_container_width=True)
            
            # Performance rating display
            if prediction == 1:
                result_class = "excellent-performance"
                emoji = "🌟"
                rating_text = "Excellent Performance"
                color = "#10b981"
                recommendation = """
                **Key Strengths:**
                - Demonstrates exceptional work quality
                - Consistently meets or exceeds targets
                - Shows strong potential for growth
                
                **Recommendations:**
                - Consider for leadership roles
                - Provide challenging projects
                - Mentor junior team members
                """
            else:
                result_class = "needs-improvement"
                emoji = "📈"
                rating_text = "Needs Improvement"
                color = "#fb923c"
                recommendation = """
                **Areas for Development:**
                - Enhance technical skills
                - Improve time management
                - Increase productivity levels
                
                **Recommendations:**
                - Additional training programs
                - Regular performance reviews
                - Set clear improvement goals
                """
            
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h2 style='text-align: center; color: {color};'>{emoji} {rating_text}</h2>
                <p style='text-align: center; font-size: 1.2rem; color: #a8b2d1; margin-top: -10px;'>
                    Confidence: {max(probability):.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Rating': ['Needs Improvement', 'Excellent'],
                    'Probability': probability
                })
                fig_prob = px.bar(
                    prob_df, 
                    x='Rating', 
                    y='Probability',
                    color='Probability',
                    color_continuous_scale=['#fb923c', '#10b981'],
                    text=prob_df['Probability'].apply(lambda x: f'{x:.1%}')
                )
                fig_prob.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    showlegend=False,
                    height=300
                )
                fig_prob.update_traces(textposition='outside')
                st.plotly_chart(fig_prob, use_container_width=True)
            
            with col2:
                st.markdown("### 💡 Recommendations")
                st.markdown(recommendation)
    
    with tab2:
        st.markdown("### 📈 Model Performance Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "Random Forest", help="Ensemble learning method")
        with col2:
            st.metric("Estimators", "100", help="Number of decision trees")
        with col3:
            st.metric("Accuracy", "100%", help="Model accuracy on test set")
        
        st.markdown("---")
        
        # Feature importance
        st.plotly_chart(create_feature_importance_chart(model), use_container_width=True)
        
        st.info("""
        **Understanding Feature Importance:**
        
        The chart above shows which factors have the most influence on performance predictions. 
        Higher values indicate features that contribute more to the model's decisions.
        """)
    
    with tab3:
        st.markdown("### ℹ️ About the Model")
        
        st.markdown("""
        #### 🤖 Random Forest Classifier
        
        This application uses a **Random Forest** ensemble learning method for classification. 
        Random Forest combines multiple decision trees to make more accurate and stable predictions.
        
        #### 📊 Model Features
        
        The model analyzes seven key employee attributes:
        
        1. **Age** - Employee's current age
        2. **Experience Years** - Total professional experience
        3. **Department** - Work department (IT, HR, Sales, Finance)
        4. **Salary** - Annual compensation
        5. **Work Hours** - Weekly working hours
        6. **Projects Handled** - Current project load
        7. **Training Hours** - Monthly training participation
        
        #### 🎯 Performance Ratings
        
        - **0 - Needs Improvement**: Requires additional support and development
        - **1 - Excellent Performance**: Exceeds expectations, ready for advancement
        
        #### 🔧 Technical Details
        
        - **Algorithm**: Random Forest Classifier
        - **Estimators**: 100 decision trees
        - **Criterion**: Gini impurity
        - **Max Depth**: 10 levels
        - **Random State**: 42 (for reproducibility)
        
        #### 📈 Use Cases
        
        - Performance evaluation
        - Talent management
        - Training needs assessment
        - Career development planning
        - Resource allocation
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #a8b2d1;'>
            <p>Developed with ❤️ using Streamlit & Scikit-learn</p>
            <p style='font-size: 0.9rem;'>© 2026 Employee Performance Analytics</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
