# 🏢 Building Energy Efficiency Predictor

A professional web application built with Streamlit that predicts a building's heating load using a Random Forest machine learning model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## 🌟 Features

* **AI-Powered Predictions**: Uses a Random Forest Regressor to accurately estimate Heating Load based on building design parameters
* **Modern UI**: Clean, responsive design built with Streamlit for a seamless user experience
* **Interactive Inputs**:
  * Dropdown selections for categorical features
  * Number inputs for continuous measurements
* **Real-time Analytics**: Instant energy load predictions computed immediately upon adjusting inputs
* **Comprehensive Reporting**: Clear efficiency classifications (e.g., High Energy Efficiency, Moderate, Low energy) based on predictions

## 🎯 Prediction Factors

The model analyzes 8 key architectural attributes:

1. **Relative Compactness** - Measure of the building's shape compared to a reference shape
2. **Surface Area** - Total outer surface area of the building
3. **Wall Area** - Total exterior wall area
4. **Roof Area** - Total surface area of the roof
5. **Overall Height** - Total height of the building
6. **Orientation** - Direction the building faces (2-5)
7. **Glazing Area** - Total area of windows/glass on the exterior
8. **Glazing Area Distribution** - How the glazing is distributed throughout the building (0-5)

## 🚀 How to Run Locally

Follow these steps to run the application on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/partha0059/RandomForest-Energy-Efficiency-Prediction.git
   cd RandomForest-Energy-Efficiency-Prediction
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the Application:**
   Open a web browser and navigate to `http://localhost:8501`.

## 📁 Repository Structure
- `app.py`: The main Streamlit application script.
- `energy_rf_model.pkl`: The pre-trained Random Forest Regression model.
- `requirements.txt`: The list of Python dependencies required to run the app.
- `README.md`: This documentation file.
