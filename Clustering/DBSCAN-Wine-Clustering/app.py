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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Clustering Analysis",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - pure black */
    .stApp {
        background: #000000;
        background-attachment: fixed;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #e94560 0%, #c72c48 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(233, 69, 96, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(233, 69, 96, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(233, 69, 96, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(233, 69, 96, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        background: rgba(233, 69, 96, 0.2);
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
        margin: 0;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(22, 33, 62, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #c72c48 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560 0%, #c72c48 100%);
        color: white;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #e94560 0%, #c72c48 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: white;
        font-weight: 500;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Dataframe styling for visibility on black background */
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #1a1a2e !important;
    }
    
    .stDataFrame table, [data-testid="stDataFrame"] table {
        background-color: #1a1a2e !important;
        color: white !important;
    }
    
    .stDataFrame thead tr th, [data-testid="stDataFrame"] thead tr th {
        background-color: #16213e !important;
        color: white !important;
        border-color: #0f3460 !important;
    }
    
    .stDataFrame tbody tr td, [data-testid="stDataFrame"] tbody tr td {
        background-color: #1a1a2e !important;
        color: white !important;
        border-color: #0f3460 !important;
    }
    
    .stDataFrame tbody tr:hover, [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #16213e !important;
    }
    
    /* Style the dataframe container */
    div[data-testid="stDataFrameResizable"] {
        background-color: #1a1a2e !important;
    }
    
    /* Additional dataframe styling overrides */
    .dataframe {
        background-color: #1a1a2e !important;
    }
    
    .dataframe thead th {
        background-color: #16213e !important;
        color: white !important;
    }
    
    .dataframe tbody td {
        background-color: #1a1a2e !important;
        color: white !important;
    }
    
    .dataframe tbody tr {
        background-color: #1a1a2e !important;
    }
    
    .dataframe tbody tr:nth-of-type(odd) {
        background-color: #16213e !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #0f3460 !important;
    }
    
    /* Override any white backgrounds */
    .element-container div div div div {
        background-color: transparent !important;
    }
    
    /* Force all table elements to have dark styling */
    table {
        background-color: #1a1a2e !important;
        color: white !important;
    }
    
    thead {
        background-color: #16213e !important;
    }
    
    th {
        background-color: #16213e !important;
        color: white !important;
    }
    
    td {
        background-color: #1a1a2e !important;
        color: white !important;
    }
    
    tr {
        background-color: #1a1a2e !important;
    }
    
    tr:hover {
        background-color: #16213e !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🍷 Wine Clustering Analysis</h1>
    <p>DBSCAN Clustering Analysis on Wine Chemical Properties</p>
</div>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load wine clustering data"""
    try:
        df = pd.read_csv('wine_clustering_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: wine_clustering_data.csv not found. Please ensure the file is in the same directory as app.py")
        st.stop()

# Load data
df = load_data()

# Helper function to create dark-styled HTML tables
def df_to_dark_html(dataframe, table_class="dark-table"):
    """Convert dataframe to HTML table with dark styling"""
    html = f"""
    <style>
        .{table_class} {{
            width: 100%;
            border-collapse: collapse;
            background-color: #1a1a2e;
            color: #ffffff;
            font-size: 14px;
        }}
        .{table_class} th {{
            background-color: #16213e;
            color: #ffffff;
            padding: 12px;
            text-align: left;
            border: 1px solid #0f3460;
            font-weight: 600;
        }}
        .{table_class} td {{
            background-color: #1a1a2e;
            color: #ffffff;
            padding: 10px;
            border: 1px solid #0f3460;
        }}
        .{table_class} tr:nth-child(even) td {{
            background-color: #16213e;
        }}
        .{table_class} tr:hover td {{
            background-color: #0f3460;
        }}
    </style>
    <div style="overflow-x: auto;">
        <table class="{table_class}">
            <thead><tr>
    """
    
    # Add column headers (including index if it has a name)
    if dataframe.index.name:
        html += f"<th>{dataframe.index.name}</th>"
    elif isinstance(dataframe.index[0], str):
        html += "<th></th>"
    
    for col in dataframe.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    # Add rows
    for idx, row in dataframe.iterrows():
        html += "<tr>"
        # Add index cell if it's meaningful
        if isinstance(idx, str) or dataframe.index.name:
            html += f"<td style='background-color: #16213e; font-weight: 600;'>{idx}</td>"
        for val in row:
            if isinstance(val, (int, float)):
                html += f"<td>{val:.2f}</td>"
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Clustering Parameters")
    
    st.markdown("---")
    
    eps = st.slider(
        "**Epsilon (ε)**",
        min_value=0.1,
        max_value=5.0,
        value=4.0,
        step=0.1,
        help="Maximum distance between two samples for one to be considered as in the neighborhood of the other"
    )
    
    min_samples = st.slider(
        "**Minimum Samples**",
        min_value=2,
        max_value=20,
        value=2,
        step=1,
        help="Number of samples in a neighborhood for a point to be considered as a core point"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Analysis Features")
    show_3d = st.checkbox("Show 3D Visualization", value=True)
    show_stats = st.checkbox("Show Detailed Statistics", value=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(233, 69, 96, 0.1); border-radius: 10px; margin-top: 2rem;'>
        <p style='font-size: 0.8rem; margin: 0;'>💡 <b>Tip:</b> Adjust epsilon and min_samples to find optimal clusters</p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Predict Cluster", "🔬 Clustering Analysis", "📈 Data Overview", "📊 Results & Insights", "ℹ️ About"])

with tab3:
    st.markdown("## 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{df.shape[0]}</p>
            <p class="metric-label">Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{df.shape[1]}</p>
            <p class="metric-label">Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{df.isnull().sum().sum()}</p>
            <p class="metric-label">Missing Values</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{df.memory_usage(deep=True).sum() / 1024:.1f} KB</p>
            <p class="metric-label">Memory Usage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    
    # Convert to HTML with custom styling
    df_preview = df.head(10)
    
    # Create HTML table with dark styling
    html_table = """
    <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #1a1a2e;
            color: white;
            font-size: 14px;
        }
        .custom-table th {
            background-color: #16213e;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #0f3460;
            font-weight: 600;
        }
        .custom-table td {
            background-color: #1a1a2e;
            color: white;
            padding: 10px;
            border: 1px solid #0f3460;
        }
        .custom-table tr:nth-child(even) td {
            background-color: #16213e;
        }
        .custom-table tr:hover td {
            background-color: #0f3460;
        }
    </style>
    <div style="overflow-x: auto;">
        <table class="custom-table">
            <thead>
                <tr>
    """
    
    # Add headers
    for col in df_preview.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead><tbody>"
    
    # Add rows
    for idx, row in df_preview.iterrows():
        html_table += "<tr>"
        for val in row:
            html_table += f"<td>{val:.2f}</td>" if isinstance(val, (int, float)) else f"<td>{val}</td>"
        html_table += "</tr>"
    
    html_table += "</tbody></table></div>"
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    
    # Convert statistics to HTML
    df_stats = df.describe()
    
    html_stats = """
    <style>
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #1a1a2e;
            color: white;
            font-size: 14px;
        }
        .stats-table th {
            background-color: #16213e;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #0f3460;
            font-weight: 600;
        }
        .stats-table td {
            background-color: #1a1a2e;
            color: white;
            padding: 10px;
            border: 1px solid #0f3460;
        }
        .stats-table tr:nth-child(even) td {
            background-color: #16213e;
        }
        .stats-table tr:hover td {
            background-color: #0f3460;
        }
        .stats-index {
            background-color: #16213e !important;
            font-weight: 600;
        }
    </style>
    <div style="overflow-x: auto;">
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Statistic</th>
    """
    
    # Add column headers
    for col in df_stats.columns:
        html_stats += f"<th>{col}</th>"
    html_stats += "</tr></thead><tbody>"
    
    # Add rows
    for idx in df_stats.index:
        html_stats += f"<tr><td class='stats-index'>{idx}</td>"
        for val in df_stats.loc[idx]:
            html_stats += f"<td>{val:.2f}</td>"
        html_stats += "</tr>"
    
    html_stats += "</tbody></table></div>"
    st.markdown(html_stats, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature distributions
    st.markdown("### 📉 Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#1a1a2e')
        features = df.columns[:6]
        for idx, feature in enumerate(features):
            row, col = idx // 2, idx % 2
            axes[row, col].hist(df[feature], bins=20, color='#e94560', alpha=0.7, edgecolor='white')
            axes[row, col].set_title(feature.replace('_', ' ').title(), color='white', fontsize=10)
            axes[row, col].set_facecolor('#16213e')
            axes[row, col].tick_params(colors='white', labelsize=8)
            axes[row, col].spines['bottom'].set_color('white')
            axes[row, col].spines['left'].set_color('white')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#1a1a2e')
        features = df.columns[6:12]
        for idx, feature in enumerate(features):
            row, col = idx // 2, idx % 2
            axes[row, col].hist(df[feature], bins=20, color='#0f3460', alpha=0.7, edgecolor='white')
            axes[row, col].set_title(feature.replace('_', ' ').title(), color='white', fontsize=10)
            axes[row, col].set_facecolor('#16213e')
            axes[row, col].tick_params(colors='white', labelsize=8)
            axes[row, col].spines['bottom'].set_color('white')
            axes[row, col].spines['left'].set_color('white')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': 8, 'color': 'white'})
    
    ax.set_title('Feature Correlation Matrix', color='white', fontsize=16, pad=20)
    ax.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    
    st.pyplot(fig)
    plt.close()

with tab2:
    st.markdown("## 🔬 DBSCAN Clustering Analysis")
    
    # Perform clustering
    @st.cache_data
    def perform_clustering(data, eps_val, min_samples_val):
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        clusters = dbscan.fit_predict(scaled_data)
        
        # PCA for visualization
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(scaled_data)
        
        return clusters, scaled_data, pca_data, scaler
    
    clusters, scaled_data, pca_data, scaler = perform_clustering(df, eps, min_samples)
    
    # Add clusters to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Calculate metrics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    # Silhouette score (only if we have more than 1 cluster and not all noise)
    if n_clusters > 1 and n_noise < len(clusters):
        silhouette_avg = silhouette_score(scaled_data, clusters)
    else:
        silhouette_avg = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{n_clusters}</p>
            <p class="metric-label">Clusters Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{n_noise}</p>
            <p class="metric-label">Noise Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{silhouette_avg:.3f}</p>
            <p class="metric-label">Silhouette Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        coverage = ((len(clusters) - n_noise) / len(clusters)) * 100
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{coverage:.1f}%</p>
            <p class="metric-label">Coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2D Visualization
    st.markdown("### 📊 2D Cluster Visualization (PCA)")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    # Define colors for clusters
    unique_clusters = sorted(set(clusters))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster_id, color in zip(unique_clusters, colors):
        cluster_mask = clusters == cluster_id
        if cluster_id == -1:
            # Noise points
            ax.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1],
                      c='gray', marker='x', s=50, alpha=0.5, label='Noise')
        else:
            ax.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1],
                      c=[color], s=100, alpha=0.7, edgecolors='white', linewidth=0.5,
                      label=f'Cluster {cluster_id}')
    
    ax.set_xlabel('First Principal Component', color='white', fontsize=12)
    ax.set_ylabel('Second Principal Component', color='white', fontsize=12)
    ax.set_title('DBSCAN Clustering Results (2D PCA)', color='white', fontsize=16, pad=20)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # 3D Visualization
    if show_3d:
        st.markdown("### 🎨 3D Interactive Cluster Visualization")
        
        df_plot = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
        df_plot['Cluster'] = ['Noise' if c == -1 else f'Cluster {c}' for c in clusters]
        
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='Cluster',
                           title='3D DBSCAN Clustering Visualization',
                           color_discrete_sequence=px.colors.qualitative.Set3,
                           height=700)
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor='#16213e', gridcolor='#0f3460', title_font=dict(color='white')),
                yaxis=dict(backgroundcolor='#16213e', gridcolor='#0f3460', title_font=dict(color='white')),
                zaxis=dict(backgroundcolor='#16213e', gridcolor='#0f3460', title_font=dict(color='white'))
            ),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
            font=dict(color='white'),
            title_font=dict(size=20, color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## 📊 Results & Insights")
    
    if n_clusters == 0:
        st.warning("⚠️ No clusters found with current parameters. Try adjusting epsilon or min_samples.")
    else:
        # Cluster distribution
        st.markdown("### 📈 Cluster Distribution")
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, 
                     color=['gray' if x == -1 else '#e94560' for x in cluster_counts.index],
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Cluster ID', color='white', fontsize=12)
        ax.set_ylabel('Number of Samples', color='white', fontsize=12)
        ax.set_title('Sample Distribution Across Clusters', color='white', fontsize=16, pad=20)
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels(['Noise' if x == -1 else f'C{x}' for x in cluster_counts.index])
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, axis='y', color='white')
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        if show_stats:
            st.markdown("### 📋 Cluster Statistics")
            
            # Calculate mean values for each cluster
            cluster_stats = df_clustered.groupby('Cluster').mean()
            
            # Display heatmap of cluster characteristics
            fig, ax = plt.subplots(figsize=(14, max(6, n_clusters * 0.8)))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            
            sns.heatmap(cluster_stats.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=cluster_stats.values.mean(), linewidths=1, 
                       cbar_kws={"shrink": 0.8}, ax=ax,
                       annot_kws={'size': 8, 'color': 'white'})
            
            ax.set_title('Average Feature Values by Cluster', color='white', fontsize=16, pad=20)
            ax.set_xlabel('Cluster', color='white', fontsize=12)
            ax.set_ylabel('Features', color='white', fontsize=12)
            ax.tick_params(colors='white')
            plt.yticks(rotation=0, color='white')
            plt.xticks(rotation=0, color='white')
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Detailed cluster information
            st.markdown("### 🔍 Detailed Cluster Information")
            
            for cluster_id in sorted(set(clusters)):
                if cluster_id == -1:
                    continue
                
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id].drop('Cluster', axis=1)
                
                with st.expander(f"**Cluster {cluster_id}** ({len(cluster_data)} samples)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Mean Values")
                        st.markdown(df_to_dark_html(cluster_data.mean().to_frame('Mean'), f'cluster-{cluster_id}-mean'), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Standard Deviation")
                        st.markdown(df_to_dark_html(cluster_data.std().to_frame('Std Dev'), f'cluster-{cluster_id}-std'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if silhouette_avg > 0.5:
            st.success(f"✅ Good clustering quality (Silhouette Score: {silhouette_avg:.3f}). The clusters are well-separated.")
        elif silhouette_avg > 0.3:
            st.info(f"ℹ️ Moderate clustering quality (Silhouette Score: {silhouette_avg:.3f}). Consider adjusting parameters for better separation.")
        else:
            st.warning(f"⚠️ Low clustering quality (Silhouette Score: {silhouette_avg:.3f}). Try different epsilon or min_samples values.")
        
        if n_noise > len(clusters) * 0.1:
            st.warning(f"⚠️ High number of noise points ({n_noise} samples, {(n_noise/len(clusters)*100):.1f}%). Consider decreasing epsilon or min_samples.")
        
        if n_clusters == 1:
            st.info("ℹ️ Only one cluster found. Try decreasing epsilon for more granular clustering.")

with tab1:
    st.markdown("## 🎯 Predict Wine Cluster")
    
    st.markdown("""
    <div class="glass-card">
        <p style="font-size: 1.1rem;">Enter the chemical properties of a wine sample to predict which cluster it belongs to.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Wine Features")
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Basic Properties")
        alcohol = st.number_input("🍷 Alcohol", min_value=10.0, max_value=16.0, value=13.0, step=0.1,
                                  help="Alcohol content in the wine")
        malic_acid = st.number_input("🔬 Malic Acid", min_value=0.0, max_value=6.0, value=2.0, step=0.1,
                                      help="Malic acid content")
        ash = st.number_input("⚗️ Ash", min_value=1.0, max_value=4.0, value=2.4, step=0.1,
                             help="Ash content")
        ash_alcanity = st.number_input("🧪 Ash Alkalinity", min_value=10.0, max_value=30.0, value=18.0, step=0.5,
                                        help="Alkalinity of the ash")
        magnesium = st.number_input("⚛️ Magnesium", min_value=70, max_value=170, value=100, step=1,
                                     help="Magnesium content")
    
    with col2:
        st.markdown("#### Phenolic Compounds")
        total_phenols = st.number_input("🌿 Total Phenols", min_value=0.0, max_value=4.0, value=2.5, step=0.1,
                                        help="Total phenolic content")
        flavanoids = st.number_input("🍇 Flavanoids", min_value=0.0, max_value=6.0, value=2.0, step=0.1,
                                     help="Flavanoid content")
        nonflavanoid_phenols = st.number_input("🧬 Nonflavanoid Phenols", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                                               help="Nonflavanoid phenolic content")
        proanthocyanins = st.number_input("💜 Proanthocyanins", min_value=0.0, max_value=4.0, value=1.5, step=0.1,
                                          help="Proanthocyanin content")
    
    with col3:
        st.markdown("#### Color & Other Properties")
        color_intensity = st.number_input("🎨 Color Intensity", min_value=1.0, max_value=15.0, value=5.0, step=0.5,
                                          help="Color intensity of the wine")
        hue = st.number_input("🌈 Hue", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                             help="Hue of the wine")
        od280 = st.number_input("📏 OD280/OD315", min_value=1.0, max_value=5.0, value=2.5, step=0.1,
                               help="OD280/OD315 ratio")
        proline = st.number_input("🧫 Proline", min_value=200, max_value=2000, value=700, step=10,
                                  help="Proline content")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Cluster", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'alcohol': [alcohol],
            'malic_acid': [malic_acid],
            'ash': [ash],
            'ash_alcanity': [ash_alcanity],
            'magnesium': [magnesium],
            'total_phenols': [total_phenols],
            'flavanoids': [flavanoids],
            'nonflavanoid_phenols': [nonflavanoid_phenols],
            'proanthocyanins': [proanthocyanins],
            'color_intensity': [color_intensity],
            'hue': [hue],
            'od280': [od280],
            'proline': [proline]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Use euclidean distance to find nearest neighbor and assign its cluster
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(input_scaled, scaled_data)
        nearest_idx = distances.argmin()
        predicted_cluster = clusters[nearest_idx]
        
        # Display result
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        if predicted_cluster == -1:
            st.markdown(f"""
            <div class="metric-container" style="background: rgba(244, 67, 54, 0.2); border-color: rgba(244, 67, 54, 0.5);">
                <p class="metric-value" style="color: #f44336;">NOISE</p>
                <p class="metric-label">This wine sample doesn't fit well into any cluster</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ The wine sample is classified as **noise/outlier**. It doesn't belong to any well-defined cluster with the current DBSCAN parameters.")
        else:
            st.markdown(f"""
            <div class="metric-container" style="background: rgba(76, 175, 80, 0.2); border-color: rgba(76, 175, 80, 0.5);">
                <p class="metric-value" style="color: #4CAF50;">Cluster {predicted_cluster}</p>
                <p class="metric-label">Predicted Cluster</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"✅ The wine sample belongs to **Cluster {predicted_cluster}** based on its chemical properties.")
        
        # Show similar wines
        st.markdown("---")
        st.markdown("### 🔍 Most Similar Wines in Dataset")
        
        # Get indices of 5 nearest wines
        similar_indices = distances[0].argsort()[:5]
        similar_wines = df.iloc[similar_indices].copy()
        similar_wines['Distance'] = distances[0][similar_indices]
        similar_wines['Cluster'] = clusters[similar_indices]
        
        st.markdown(df_to_dark_html(similar_wines, 'similar-wines-table'), unsafe_allow_html=True)
        
        # Show cluster characteristics if not noise
        if predicted_cluster != -1:
            st.markdown("---")
            st.markdown(f"### 📊 Cluster {predicted_cluster} Characteristics")
            
            cluster_data = df_clustered[df_clustered['Cluster'] == predicted_cluster].drop('Cluster', axis=1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Average Values")
                st.markdown(df_to_dark_html(cluster_data.mean().to_frame('Mean'), 'cluster-mean-table'), unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Your Input vs Cluster Average")
                comparison = pd.DataFrame({
                    'Your Input': input_data.iloc[0],
                    'Cluster Avg': cluster_data.mean()
                })
                st.markdown(df_to_dark_html(comparison, 'comparison-table'), unsafe_allow_html=True)

with tab5:
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application performs **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** clustering analysis 
    on wine chemical properties to identify natural groupings based on their chemical composition.
    
    ---
    
    ### 📚 Dataset Information
    
    The wine clustering dataset includes the following features:
    
    - **Alcohol**: Alcohol content in the wine
    - **Malic Acid**: Malic acid content
    - **Ash**: Ash content
    - **Ash Alkalinity**: Alkalinity of the ash
    - **Magnesium**: Magnesium content
    - **Total Phenols**: Total phenolic content
    - **Flavanoids**: Flavanoid content
    - **Nonflavanoid Phenols**: Nonflavanoid phenolic content
    - **Proanthocyanins**: Proanthocyanin content
    - **Color Intensity**: Color intensity of the wine
    - **Hue**: Hue of the wine
    - **OD280**: OD280/OD315 ratio
    - **Proline**: Proline content
    
    **Dataset Credits**: UCI Machine Learning Repository
    
    ---
    
    ### 🔬 Methodology
    
    1. **Data Preprocessing**: Features are standardized using StandardScaler to ensure equal contribution
    2. **DBSCAN Clustering**: Applied with user-defined epsilon and min_samples parameters
    3. **Dimensionality Reduction**: PCA is used to visualize high-dimensional clusters in 2D/3D space
    4. **Quality Assessment**: Silhouette score measures the quality of clustering
    
    ---
    
    ### ⚙️ DBSCAN Parameters
    
    - **Epsilon (ε)**: Maximum distance between two samples for one to be considered in the neighborhood of the other
    - **Min Samples**: Minimum number of samples in a neighborhood for a point to be considered as a core point
    
    ---
    
    ### 👨‍💻 Developer Information
    
    **Created by**: Partha Sarathi R  
    **Purpose**: Machine Learning Assignment - Unsupervised Learning  
    **Algorithm**: DBSCAN Clustering  
    **Framework**: Streamlit  
    
    ---
    
    ### 🚀 How to Use
    
    1. Adjust **Epsilon** and **Min Samples** in the sidebar
    2. Explore the **Data Overview** tab to understand the dataset
    3. View clustering results in the **Clustering Analysis** tab
    4. Analyze detailed insights in the **Results & Insights** tab
    5. Experiment with different parameter values to find optimal clusters
    
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>🍷 Wine Clustering Analysis | DBSCAN Algorithm</p>
    <p>Created by <b>Partha Sarathi R</b> | Machine Learning Assignment</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">Built with Streamlit, Scikit-learn, and Plotly</p>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
