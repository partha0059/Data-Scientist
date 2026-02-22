# 🫀 AI Health Risk Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Lasso%20%26%20Linear%20Regression-8B5CF6?style=for-the-badge"/>
</p>

<p align="center">
  A professional, AI-powered health risk assessment tool built with <strong>Streamlit</strong>, using <strong>Lasso & Linear Regression</strong> to predict a patient's overall health risk score based on key clinical indicators.
</p>

---

## ✨ Features

- 🩺 **Clinical Input Panel** — Collects 11 patient health parameters across 3 categories  
- 🤖 **ML Prediction** — Pre-trained Lasso/Linear Regression model delivers instant risk scores  
- 📊 **Risk Categorisation** — Classifies results as Low / Moderate / High with visual feedback  
- 💎 **Dark Glassmorphism UI** — Modern, professional interface with gradient accents and animated cards  
- 📈 **Summary Metrics** — Quick-glance metric cards for key vitals after every prediction  

---

## 🖥️ Live Demo

> Deploy this app on **Streamlit Community Cloud** for free.  
> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/partha0059/Lasso_Linear.git
cd Lasso_Linear
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
Lasso_Linear/
│
├── app.py              # Streamlit application (main UI + prediction logic)
├── model.pkl           # Pre-trained Lasso / Linear Regression model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🧠 Input Features

| Feature | Description | Range |
|---|---|---|
| **Age** | Patient age in years | 0 – 120 |
| **BMI** | Body Mass Index | 10.0 – 60.0 |
| **Blood Pressure** | Systolic blood pressure (mmHg) | 50 – 200 |
| **Cholesterol** | Total cholesterol (mg/dL) | 100 – 400 |
| **Glucose Level** | Fasting glucose (mg/dL) | 50 – 300 |
| **Insulin Level** | Serum insulin (µU/mL) | 0 – 500 |
| **Heart Rate** | Resting heart rate (bpm) | 40 – 200 |
| **Activity Level** | Physical activity score (1–10) | 1 – 10 |
| **Diet Quality** | Diet quality score (1–10) | 1 – 10 |
| **Smoking Status** | Whether the patient smokes | Yes / No |
| **Alcohol Intake** | Weekly alcohol units | 0 – 50 |

---

## 📊 Risk Score Interpretation

| Score Range | Risk Level | Recommendation |
|---|---|---|
| **< 33** | 🟢 Low Risk | Maintain current healthy lifestyle |
| **33 – 65** | 🟡 Moderate Risk | Lifestyle improvements recommended |
| **≥ 66** | 🔴 High Risk | Medical consultation strongly advised |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS (Glassmorphism) |
| **ML Model** | Scikit-learn – Lasso / Linear Regression |
| **Data Processing** | NumPy, Pandas |
| **Model Persistence** | Joblib |

---

## ⚠️ Disclaimer

> This application is intended **for educational and research purposes only**.  
> It does **not** constitute medical advice or a clinical diagnosis.  
> Always consult a qualified healthcare professional for medical decisions.

---

## 👨‍💻 Developer

<table>
  <tr>
    <td><strong>Name</strong></td>
    <td>Partha Sarathi R</td>
  </tr>
  <tr>
    <td><strong>Project</strong></td>
    <td>AI Health Risk Predictor — Lasso &amp; Linear Regression</td>
  </tr>
  <tr>
    <td><strong>GitHub</strong></td>
    <td><a href="https://github.com/partha0059">@partha0059</a></td>
  </tr>
</table>

---

<p align="center">
  Made with ❤️ using <strong>Streamlit</strong> &nbsp;|&nbsp; © 2026 Partha Sarathi R
</p>