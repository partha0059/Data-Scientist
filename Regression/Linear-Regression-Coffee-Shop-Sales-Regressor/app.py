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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(
    page_title="Coffee Shop Analytics - Partha Sarathi R",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK NEON THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }

    /* Sidebar Dark Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    p, label, div[data-testid="stMarkdownContainer"] {
        color: #d1d5db !important;
    }

    /* Neon Purple Button */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border: none;
        padding: 16px 28px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.5);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.7);
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 2em !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-weight: 600 !important;
    }

    /* Input Fields */
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
    }
    
    .stNumberInput>div>div>input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 4px;
    }

    /* Alert Boxes */
    .stAlert {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        color: #e5e7eb !important;
    }
    
    /* Success Box */
    .element-container:has(.stSuccess) {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('Coffee_Sales_model.pkl')
        df = pd.read_excel("coffee_shop_sales_dataset.xlsx")
        return model, df
    except Exception as e:
        st.error(f"⚠️ Error loading resources: {e}")
        return None, None

model, df = load_resources()

# --- Main App ---
if model is not None and df is not None:
    
    # --- Header ---
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; background: rgba(30, 41, 59, 0.4); border-radius: 16px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 25px;'>
        <h1 style='font-size: 2.8em; margin-bottom: 8px; background: linear-gradient(90deg, #60a5fa, #a78bfa, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ☕ Coffee Shop Analytics
        </h1>
        <p style='font-size: 1.1em; color: #9ca3af; font-weight: 600; margin: 5px 0;'>
            AI-Powered Revenue Intelligence Platform
        </p>
        <p style='font-size: 0.95em; color: #6b7280; margin-top: 8px;'>
            Data Science Project by <span style='color: #a78bfa; font-weight: 700;'>Partha Sarathi R</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("# 📊 Analytics Controls")
        
        st.markdown("""
        <div style='background: rgba(99, 102, 241, 0.1); padding: 12px; border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 20px;'>
            <p style='margin: 0; font-size: 0.9em; color: #c7d2fe;'>🎯 Adjust the 6 model parameters below</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💰 Cost Parameters")
        
        # Ingredient Cost
        st.markdown("**🥦 Ingredient Cost ($)**")
        col1, col2 = st.columns([3, 2])
        with col1:
            ing_slider = st.slider("", float(df['Ingredient_Cost'].min()), float(df['Ingredient_Cost'].max()), 
                                  float(df['Ingredient_Cost'].mean()), 5.0, key='ing_s', label_visibility="collapsed")
        with col2:
            ing_input = st.number_input("", float(df['Ingredient_Cost'].min()), float(df['Ingredient_Cost'].max()), 
                                       ing_slider, 5.0, key='ing_i', label_visibility="collapsed")
        ingredient_cost = ing_input
        st.markdown("---")
        
        # Total Costs
        st.markdown("**📉 Total Costs ($)**")
        col1, col2 = st.columns([3, 2])
        with col1:
            tot_slider = st.slider("", float(df['Total_Costs'].min()), float(df['Total_Costs'].max()), 
                                  float(df['Total_Costs'].mean()), 10.0, key='tot_s', label_visibility="collapsed")
        with col2:
            tot_input = st.number_input("", float(df['Total_Costs'].min()), float(df['Total_Costs'].max()), 
                                       tot_slider, 10.0, key='tot_i', label_visibility="collapsed")
        total_costs = tot_input
        st.markdown("---")
        
        st.markdown("### 📊 Sales Metrics")
        
        # Coffee Sales
        st.markdown("**☕ Coffee Sales ($)**")
        col1, col2 = st.columns([3, 2])
        with col1:
            cof_slider = st.slider("", float(df['Coffee_Sales'].min()), float(df['Coffee_Sales'].max()), 
                                  float(df['Coffee_Sales'].mean()), 10.0, key='cof_s', label_visibility="collapsed")
        with col2:
            cof_input = st.number_input("", float(df['Coffee_Sales'].min()), float(df['Coffee_Sales'].max()), 
                                       cof_slider, 10.0, key='cof_i', label_visibility="collapsed")
        coffee_sales = cof_input
        st.markdown("---")
        
        # Pastry Sales
        st.markdown("**🥐 Pastry Sales ($)**")
        col1, col2 = st.columns([3, 2])
        with col1:
            pas_slider = st.slider("", float(df['Pastry_Sales'].min()), float(df['Pastry_Sales'].max()), 
                                  float(df['Pastry_Sales'].mean()), 10.0, key='pas_s', label_visibility="collapsed")
        with col2:
            pas_input = st.number_input("", float(df['Pastry_Sales'].min()), float(df['Pastry_Sales'].max()), 
                                       pas_slider, 10.0, key='pas_i', label_visibility="collapsed")
        pastry_sales = pas_input
        st.markdown("---")
        
        # Customers
        st.markdown("**👥 No of Customers**")
        col1, col2 = st.columns([3, 2])
        with col1:
            cus_slider = st.slider("", float(df['Num_Customers'].min()), float(df['Num_Customers'].max()), 
                                  float(df['Num_Customers'].mean()), 1.0, key='cus_s', label_visibility="collapsed")
        with col2:
            cus_input = st.number_input("", float(df['Num_Customers'].min()), float(df['Num_Customers'].max()), 
                                       cus_slider, 1.0, key='cus_i', label_visibility="collapsed")
        num_customers = cus_input
        st.markdown("---")
        
        st.markdown("### 💎 Financial Target")
        
        # Daily Profit
        st.markdown("**💰 Daily Profit ($)**")
        col1, col2 = st.columns([3, 2])
        with col1:
            pro_slider = st.slider("", float(df['Daily_Profit'].min()), float(df['Daily_Profit'].max()), 
                                  float(df['Daily_Profit'].mean()), 10.0, key='pro_s', label_visibility="collapsed")
        with col2:
            pro_input = st.number_input("", float(df['Daily_Profit'].min()), float(df['Daily_Profit'].max()), 
                                       pro_slider, 10.0, key='pro_i', label_visibility="collapsed")
        daily_profit = pro_input
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 16px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-top: 20px;">
            <p style="font-size: 0.85em; color: #a78bfa; margin: 2px 0; font-weight: 600;">Powered by AI</p>
            <p style="font-size: 1.15em; font-weight: 800; margin: 5px 0; color: #c4b5fd;">Partha Sarathi R</p>
            <p style="font-size: 0.75em; color: #818cf8; margin: 2px 0;">Data Science Project</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Prediction Button ---
    if st.button("🚀 Generate Prediction", use_container_width=True):
        
        input_data = np.array([[ingredient_cost, coffee_sales, num_customers, daily_profit, pastry_sales, total_costs]])
        prediction = max(0, model.predict(input_data)[0])
        
        # Metrics
        margin = (daily_profit / prediction * 100) if prediction > 0 else 0
        roi = ((prediction - total_costs) / total_costs * 100) if total_costs > 0 else 0
        avg_spend = (prediction / num_customers) if num_customers > 0 else 0
        
        # --- Metric Cards ---
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div style='background: rgba(34, 197, 94, 0.15); padding: 20px; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.3);'>
                <p style='color: #86efac; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;'>PREDICTED REVENUE</p>
                <p style='color: #22c55e; font-size: 2em; font-weight: 800; margin: 0;'>${prediction:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div style='background: rgba(249, 115, 22, 0.15); padding: 20px; border-radius: 12px; border: 1px solid rgba(249, 115, 22, 0.3);'>
                <p style='color: #fdba74; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;'>PROFIT MARGIN</p>
                <p style='color: #f97316; font-size: 2em; font-weight: 800; margin: 0;'>{margin:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div style='background: rgba(59, 130, 246, 0.15); padding: 20px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);'>
                <p style='color: #93c5fd; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;'>ROI</p>
                <p style='color: #3b82f6; font-size: 2em; font-weight: 800; margin: 0;'>{roi:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            st.markdown(f"""
            <div style='background: rgba(236, 72, 153, 0.15); padding: 20px; border-radius: 12px; border: 1px solid rgba(236, 72, 153, 0.3);'>
                <p style='color: #f9a8d4; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;'>AVG CUSTOMER SPEND</p>
                <p style='color: #ec4899; font-size: 2em; font-weight: 800; margin: 0;'>${avg_spend:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- Vibrant Charts ---
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("### 🌈 Revenue Composition")
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Predicted Revenue', 'Total Costs', 'Daily Profit'],
                values=[prediction, total_costs, abs(daily_profit)],
                hole=.5,
                marker_colors=['#06b6d4', '#ec4899', '#22c55e'],
                textinfo='label+percent',
                textfont=dict(size=13, color='white', family='Inter'),
                hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<extra></extra>'
            )])
            
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb', size=12),
                height=350,
                showlegend=True,
                legend=dict(font=dict(color='#e5e7eb'))
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with viz_col2:
            st.markdown("### 📊 Profit Efficiency %")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(0, min(100, margin)),
                title={'text': "", 'font': {'color': '#e5e7eb', 'size': 16}},
                number={'font': {'color': '#a78bfa', 'size': 50}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#6b7280"},
                    'bar': {'color': "#a78bfa", 'thickness': 0.75},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(139, 92, 246, 0.3)",
                    'steps': [
                        {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [25, 50], 'color': 'rgba(251, 191, 36, 0.3)'},
                        {'range': [50, 75], 'color': 'rgba(59, 130, 246, 0.3)'},
                        {'range': [75, 100], 'color': 'rgba(34, 197, 94, 0.3)'}
                    ]
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "#e5e7eb"},
                height=350
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # --- Colorful Bar Chart ---
        st.markdown("### 🎨 Colorful Sales Analysis")
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Coffee<br>Sales', 'Pastry<br>Sales', 'Ingredient<br>Cost', 'Total<br>Costs', 'Daily<br>Profit', 'Predicted<br>Revenue'],
                y=[coffee_sales, pastry_sales, ingredient_cost, total_costs, daily_profit, prediction],
                marker=dict(
                    color=['#06b6d4', '#a855f7', '#fb923c', '#f43f5e', '#22c55e', '#3b82f6'],
                    line=dict(color='#1e293b', width=2)
                ),
                text=[f'${coffee_sales:,.0f}', f'${pastry_sales:,.0f}', f'${ingredient_cost:,.0f}', 
                      f'${total_costs:,.0f}', f'${daily_profit:,.0f}', f'${prediction:,.0f}'],
                textposition='outside',
                textfont=dict(color='#e5e7eb', size=14, family='Inter'),
                hovertemplate='<b>%{x}</b><br>$%{y:,.2f}<extra></extra>'
            )
        ])
        
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            font=dict(color='#e5e7eb', size=13),
            xaxis=dict(showgrid=False, color='#9ca3af'),
            yaxis=dict(showgrid=True, gridcolor='rgba(75, 85, 99, 0.3)', color='#9ca3af', title='Amount ($)'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # --- Customer Insights Radar ---
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 💸 Financial Waterfall")
            
            fig_waterfall = go.Figure(go.Waterfall(
                x=['Revenue', 'Ingredients', 'Staff', 'Utilities', 'Rent', 'Net Profit'],
                y=[prediction, -ingredient_cost, -total_costs*0.3, -total_costs*0.1, -total_costs*0.1, daily_profit],
                measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
                text=[f'${prediction:,.0f}', f'-${ingredient_cost:,.0f}', f'-${total_costs*0.3:,.0f}',
                      f'-${total_costs*0.1:,.0f}', f'-${total_costs*0.1:,.0f}', f'${daily_profit:,.0f}'],
                textposition='outside',
                textfont=dict(color='#e5e7eb', size=12),
                connector=dict(line=dict(color='rgba(139, 92, 246, 0.3)')),
                increasing=dict(marker=dict(color='#06b6d4')),
                decreasing=dict(marker=dict(color='#ec4899')),
                totals=dict(marker=dict(color='#a78bfa'))
            ))
            
            fig_waterfall.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.3)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(color='#9ca3af'),
                yaxis=dict(showgrid=True, gridcolor='rgba(75, 85, 99, 0.3)', color='#9ca3af'),
                height=400
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col_b:
            st.markdown("### 🎯 Customer Insights")
            
            categories = ['Revenue', 'Profit', 'Customers', 'Coffee', 'Pastry']
            values = [
                (prediction / df['Daily_Revenue'].max() * 100),
                (daily_profit / df['Daily_Profit'].max() * 100),
                (num_customers / df['Num_Customers'].max() * 100),
                (coffee_sales / df['Coffee_Sales'].max() * 100),
                (pastry_sales / df['Pastry_Sales'].max() * 100)
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(139, 92, 246, 0.3)',
                line=dict(color='#a78bfa', width=3),
                marker=dict(size=10, color='#ec4899'),
                name='Current'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(30, 41, 59, 0.3)',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        showgrid=True,
                        gridcolor='rgba(75, 85, 99, 0.3)',
                        color='#9ca3af'
                    ),
                    angularaxis=dict(
                        color='#e5e7eb',
                        gridcolor='rgba(75, 85, 99, 0.3)'
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb', size=12),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.success("✨ Prediction completed successfully!")
    
    else:
        # --- Landing State ---
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: rgba(30, 41, 59, 0.4); border-radius: 16px; border: 1px solid rgba(139, 92, 246, 0.3);'>
            <h2 style='color: #a78bfa; margin-bottom: 15px;'>🎯 Ready to Analyze</h2>
            <p style='color: #9ca3af; font-size: 1.05em;'>Adjust the parameters in the sidebar and click "Generate Prediction" to see AI-powered insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Dataset Preview")
        
        sample_data = df.head(20).reset_index(drop=True)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=sample_data.index,
            y=sample_data['Daily_Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color='#a78bfa'),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        ))
        
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.3)',
            font=dict(color='#e5e7eb'),
            xaxis=dict(showgrid=False, color='#9ca3af', title='Day'),
            yaxis=dict(showgrid=True, gridcolor='rgba(75, 85, 99, 0.3)', color='#9ca3af', title='Revenue ($)'),
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.error("⚠️ Failed to load model or dataset.")


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
