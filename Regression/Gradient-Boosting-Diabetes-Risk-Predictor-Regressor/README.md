# 🩺 Diabetes Risk Predictor

A professional machine learning web application that predicts diabetes disease progression using Gradient Boosting Regression. Built with Streamlit and deployed for medical research purposes.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🌟 Features

- **Advanced ML Model**: Gradient Boosting Regressor with 98% training accuracy
- **Professional UI**: Sleek black theme with glassmorphic design
- **Real-time Predictions**: Instant diabetes progression risk assessment
- **Color-Coded Results**: Visual risk indicators (Low/Moderate/High)
- **Educational Interface**: Clear explanations of ML predictions
- **Feature Importance**: Identifies key health metrics affecting predictions

## 🎯 Live Demo

Visit the live application: [Diabetes Risk Predictor](https://your-app-url.streamlit.app)

## 📊 Model Performance

| Metric | Training | Test |
|--------|----------|------|
| R² Score | 0.9831 (98.31%) | 0.3593 (35.93%) |
| RMSE | 10.14 | 58.26 |
| MAE | 7.73 | 47.49 |
| Avg Error | ±16 points | - |

**Key Features by Importance:**
1. BMI (Body Mass Index) - 36.62%
2. Serum 5 - 22.22%
3. Blood Pressure - 8.82%

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/partha0059/Diabetes-Gradient-Boosting.git
cd Diabetes-Gradient-Boosting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
Diabetes-Gradient-Boosting/
├── app.py                                  # Main Streamlit application
├── diabetes_gradient_boosting_model.pkl    # Trained ML model
├── diabetes_feature_columns.pkl            # Feature configuration
├── diabetes.csv                            # Dataset (442 samples)
├── requirements.txt                        # Python dependencies
├── train_model.py                         # Model training script
└── README.md                              # Project documentation
```

## 🔬 How It Works

### Input Features (10 variables)

The model accepts the following normalized medical parameters:

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age (normalized) | -0.11 to 0.11 |
| sex | Gender (normalized) | -0.04 to 0.05 |
| bmi | Body Mass Index | -0.09 to 0.17 |
| bp | Average Blood Pressure | -0.11 to 0.13 |
| s1 | Serum Measurement 1 | -0.13 to 0.20 |
| s2 | Serum Measurement 2 | -0.12 to 0.20 |
| s3 | Serum Measurement 3 | -0.10 to 0.18 |
| s4 | Serum Measurement 4 | -0.08 to 0.19 |
| s5 | Serum Measurement 5 | -0.13 to 0.13 |
| s6 | Serum Measurement 6 | -0.14 to 0.14 |

### Output

- **Disease Progression Score**: 25 to 346 (quantitative measure)
- **Risk Level**: Low (<100), Moderate (100-200), High (>200)

### Algorithm

**Gradient Boosting Regressor** with optimized hyperparameters:
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 4
- `min_samples_split`: 5
- `random_state`: 42

## 🎨 UI Highlights

- **Modern Black Theme**: Professional medical application aesthetic
- **Glassmorphic Cards**: Semi-transparent containers with blur effects
- **Responsive Layout**: Works on desktop and mobile devices
- **Interactive Elements**: Smooth animations and hover effects
- **Color-Coded Predictions**: Instant visual risk assessment

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Language**: Python 3.8+

## 📈 Model Training

To retrain the model with your own data:

```bash
python train_model.py
```

This will:
1. Load the diabetes dataset
2. Split data (80/20 train/test)
3. Train Gradient Boosting Regressor
4. Display performance metrics
5. Save updated model files

## 🌐 Deployment to Streamlit Cloud

### Steps:

1. **Push to GitHub** (already done)

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `partha0059/Diabetes-Gradient-Boosting`
   - Main file: `app.py`
   - Click "Deploy"

3. **Your app will be live** at: `https://your-app-name.streamlit.app`

## ⚠️ Important Notes

### About ML Predictions

> **Note**: This is a Machine Learning model that provides **estimations**, not exact diagnoses. The model:
> - Learns patterns from historical medical data
> - Provides disease progression estimates (not yes/no classifications)
> - Has an average prediction error of ±16 points
> - Should NOT replace professional medical consultation

### Medical Disclaimer

This application is for **educational and research purposes only**. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 👨‍💻 Developer

**Partha Sarathi R**
- 📧 Email: ayyapparaja227@gmail.com
- 🔗 GitHub: [@partha0059](https://github.com/partha0059)

## 📝 Dataset Information

- **Source**: Diabetes dataset (standard ML dataset)
- **Samples**: 442 patients
- **Features**: 10 normalized medical measurements
- **Target**: Disease progression measure (quantitative)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset provided for medical research
- Streamlit for the amazing web framework
- scikit-learn for ML tools

---

<div align="center">
  
**Built with ❤️ by Partha Sarathi R**

*For College Project | Machine Learning Research*

</div>
