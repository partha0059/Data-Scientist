# 🩺 Breast Cancer Classification using KNN

A professional web application built with Streamlit that predicts whether a breast mass is benign or malignant using a pre-trained K-Nearest Neighbors (KNN) machine learning model.

## 🌟 Features

- **AI-Powered Predictions**: Uses a K-Nearest Neighbors (KNN) classifier to accurately predict breast cancer diagnosis based on 30 geometric features of cell nuclei.
- **Modern UI**: Clean, responsive, and user-friendly interface built with Streamlit, featuring custom styling.
- **Interactive Inputs**: Allows users to input 30 different cell nucleus features (e.g., radius, texture, perimeter) separated into dynamic columns.
- **Real-time Analytics**: Instant diagnosis predictions computed instantly upon user submission.
- **Clear Results**: Visual feedback indicating whether the diagnosis is Benign (Non-Cancerous) or Malignant (Cancerous).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/partha0059/KNN-Breast-Cancer-Classification.git
   cd KNN-Breast-Cancer-Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

### Running the App

Run the following command to start the Streamlit application locally:
```bash
streamlit run app.py
```

## 🛠️ Technology Stack

- **Frontend/Backend Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (K-Nearest Neighbors approach)
- **Data Manipulation**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

## 📂 Repository Contents

- `app.py`: The main Streamlit web application script.
- `knn_model.pkl`: The pre-trained K-Nearest Neighbors model used for inference.
- `scaler.pkl`: The fitted StandardScaler used to standardize feature inputs.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation.
