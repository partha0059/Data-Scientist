# 🛍️ K-Means Customer Segmentation

A modern, interactive web application for customer segmentation using K-Means clustering algorithm. Built with Streamlit, featuring a professional dark glassmorphic UI with stunning animations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![ML](https://img.shields.io/badge/ML-K--Means-green.svg)

## 📋 Overview

This application segments customers into 5 distinct groups based on their **Annual Income** and **Spending Score** using the K-Means clustering algorithm. Each segment is uniquely identified with color-coded visualizations and business insights.

### Customer Segments

| Group | Type | Characteristics | Strategy |
|-------|------|-----------------|----------|
| 👑 **Group 4** | Elite/Target | High Income, High Spending | Premium offers and loyalty programs |
| 💎 **Group 3** | Frugal/Target | High Income, Low Spending | Upsell value proposition |
| ⭐ **Group 2** | Standard Customer | Average Income, Average Spending | Maintain engagement |
| ⚠️ **Group 1** | Careless Spender | Low Income, High Spending | Impulse deals (Credit risk) |
| 🎯 **Group 0** | Sensible Customer | Low Income, Low Spending | Focus on value and discounts |

## ✨ Features

- 🎨 **Professional Dark Theme** - Sleek black background with neon accents
- 💎 **Glassmorphic Design** - Modern glass-effect UI components
- ✨ **Subtle Animations** - Professional hover effects and transitions
- 🎯 **Real-time Predictions** - Instant customer segment classification
- 📊 **Cluster Analysis** - Visual explanation of centroid distances
- 🌈 **Color-Coded Results** - Unique styling for each customer segment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/partha0059/K-mean-Customer-Segmentation.git
cd K-mean-Customer-Segmentation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (First time only)
```bash
python setup_model.py
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
```
Navigate to: http://localhost:8501
```

## 📁 Project Structure

```
K-mean-Customer-Segmentation/
├── app.py                 # Main Streamlit application
├── style.css             # Custom CSS styling with animations
├── setup_model.py        # Model training script
├── Mall_Customers.csv    # Dataset (200 customer records)
├── kmeans_model.pkl      # Trained K-Means model
├── scaler.joblib.pkl     # Feature scaler
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🎯 How It Works

### 1. Data Processing
- Customer data includes Annual Income (k$) and Spending Score (1-100)
- Features are normalized using StandardScaler for optimal clustering

### 2. K-Means Clustering
- Algorithm partitions customers into 5 distinct clusters
- Each cluster represents a unique customer segment
- Centroids are calculated based on cluster characteristics

### 3. Prediction
- User inputs are scaled and compared to cluster centroids
- Closest centroid determines the customer segment
- Results displayed with business insights and recommendations

## 🛠️ Technology Stack

- **Frontend:** Streamlit, HTML, CSS
- **Backend:** Python 3.8+
- **Machine Learning:** scikit-learn (K-Means)
- **Data Processing:** pandas, numpy
- **Visualization:** Custom CSS animations

## 📊 Dataset

The application uses the **Mall Customers Dataset** containing:
- **Rows:** 200 customer records
- **Features:**
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

## 🎨 UI/UX Highlights

- **Pure Black Background** (#000000) for maximum contrast
- **Neon Blue Accents** (#00d2ff) for modern tech aesthetic
- **Gradient Headers** - Unique color gradient for each segment
- **Glittering Borders** - Subtle pulsing border animations
- **Smooth Transitions** - Professional 0.3-0.6s animations
- **Responsive Layout** - Centered design with optimal spacing

## 📸 Screenshots

### Main Interface
Clean, centered input form with professional styling

### Prediction Results
Color-coded customer segment with business insights and data visualization

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 👨‍💻 Developer

**Partha Sarathi R**

Project developed as part of machine learning coursework demonstrating practical application of unsupervised learning techniques.

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Mall Customers Dataset
- Framework: Streamlit
- Algorithm: K-Means Clustering (scikit-learn)

---

<div align="center">
Made with ❤️ for Machine Learning Education
</div>