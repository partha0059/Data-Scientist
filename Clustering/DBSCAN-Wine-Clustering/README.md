# 🍷 Wine Clustering Analysis - DBSCAN

A professional Streamlit web application for performing DBSCAN clustering analysis on wine chemical properties with modern glassmorphism UI design.

## 📋 Overview

This project implements unsupervised learning using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to identify natural groupings in wine datasets based on their chemical composition.

## ✨ Features

- **Modern UI/UX**: Dark glassmorphism design with gradient backgrounds and smooth animations
- **Interactive Parameter Tuning**: Real-time adjustment of DBSCAN parameters (epsilon and min_samples)
- **Comprehensive Visualizations**:
  - 2D PCA scatter plots with cluster coloring
  - 3D interactive cluster visualization
  - Feature distribution histograms
  - Correlation heatmaps
  - Cluster statistics heatmaps
- **Detailed Analysis**:
  - Silhouette score calculation
  - Cluster-wise statistics
  - Noise point detection
  - Coverage metrics
- **Professional Layout**: Organized tabs for data exploration, clustering, results, and information

## 📦 Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "DBSCAN Cluster"
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Ensure the dataset file is present**:
   - The application expects `wine_clustering_data.csv` in the same directory

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to the provided local URL (typically `http://localhost:8501`)

4. **Interact with the application**:
   - Adjust epsilon and min_samples in the sidebar
   - Explore different tabs for various analyses
   - Toggle 3D visualization and detailed statistics options

## 📊 Dataset

The wine clustering dataset contains 178 samples with 13 chemical properties:

- Alcohol
- Malic Acid
- Ash
- Ash Alkalinity
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280
- Proline

**Source**: UCI Machine Learning Repository

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (DBSCAN, StandardScaler, PCA)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Styling**: Custom CSS with glassmorphism effects

## 📁 Project Structure

```
DBSCAN Cluster/
├── app.py                      # Main Streamlit application
├── wine_clustering_data.csv    # Wine dataset
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── assignment (2).ipynb        # Original assignment notebook
```

## 🎯 DBSCAN Parameters

- **Epsilon (ε)**: Maximum distance between two samples for neighborhood consideration
  - Range: 0.1 - 5.0
  - Default: 0.5

- **Min Samples**: Minimum samples in a neighborhood for a core point
  - Range: 2 - 20
  - Default: 5

## 📈 Clustering Quality Metrics

The application calculates:
- **Silhouette Score**: Measures cluster separation quality (-1 to 1)
- **Number of Clusters**: Total distinct groups found
- **Noise Points**: Samples that don't belong to any cluster
- **Coverage**: Percentage of samples successfully clustered

## 🎨 Design Features

- Dark gradient background (#1a1a2e → #16213e → #0f3460)
- Wine-inspired accent color (#e94560)
- Glassmorphic cards with backdrop blur effects
- Smooth hover transitions and animations
- Professional typography (Inter font family)
- Responsive layout

## 👨‍💻 Developer

**Partha Sarathi R**

## 📝 Assignment Details

This project was developed as part of a machine learning assignment focusing on unsupervised learning techniques, specifically DBSCAN clustering for exploratory data analysis.

## 🔧 Troubleshooting

**Issue**: Application shows "wine_clustering_data.csv not found"
- **Solution**: Ensure the CSV file is in the same directory as app.py

**Issue**: Dependencies installation fails
- **Solution**: Try upgrading pip: `pip install --upgrade pip` then reinstall requirements

**Issue**: Visualizations not rendering
- **Solution**: Clear Streamlit cache: `streamlit cache clear`

## 📄 License

This project is created for educational purposes as part of a machine learning assignment.

---

**Built with ❤️ using Streamlit**
