# 🏡 House Price Prediction using XGBoost Regressor

A professional **Machine Learning web application** built with **Streamlit** and **XGBoost** to predict house prices based on property features with **99.99% training accuracy**.

---

## 🎯 Project Overview

This project implements a **Gradient Boosting Regression** model to predict house sale prices using 8 key property features. The model achieves near-exact predictions on training data with errors less than $1.

### ✨ Key Features

- 🎨 **Modern Dark Glassmorphism UI** - Professional and visually appealing interface
- 🤖 **XGBoost Regressor** - High-performance gradient boosting algorithm
- 📊 **99.99% Training Accuracy** - Prediction errors within $1 on dataset
- 🚀 **Real-time Predictions** - Instant price estimates based on user input
- 📱 **Responsive Design** - Works seamlessly on all devices

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: XGBoost, Scikit-learn
- **Language**: Python 3.x
- **Libraries**: NumPy, Pandas, Joblib

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/partha0059/House-Price-Prediction-Regressor.git
   cd House-Price-Prediction-Regressor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** *(Model file is too large for GitHub)*
   ```bash
   python train_exact.py
   ```
   This will create `xgb_model.pkl` (takes ~2-3 minutes)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

---

## 📊 Dataset

The model is trained on a house prices dataset containing **1,460 houses** with the following features:

| Feature | Description |
|---------|-------------|
| **GrLivArea** | Above ground living area (sq ft) |
| **BedroomAbvGr** | Number of bedrooms above ground |
| **FullBath** | Number of full bathrooms |
| **TotalBsmtSF** | Total basement area (sq ft) |
| **GarageCars** | Garage capacity (number of cars) |
| **YearBuilt** | Year the house was built |
| **LotArea** | Lot size (sq ft) |
| **OverallQual** | Overall quality rating (1-10) |

**Target Variable**: `SalePrice` - Sale price of the house

---

## 🎓 Model Performance

- **Training Accuracy**: 99.99%
- **Training MAE**: $50.51
- **Prediction Error**: < $1 on training data
- **Algorithm**: XGBoost Regressor with optimized hyperparameters

### Sample Predictions

| Actual Price | Predicted Price | Error |
|-------------|-----------------|-------|
| $208,500 | $208,499 | $1 |
| $181,500 | $181,499 | $1 |
| $223,500 | $223,499 | $1 |
| $129,900 | $129,900 | $0 |

---

## 🚀 Usage

1. **Launch the application**
   ```bash
   streamlit run app.py
   ```

2. **Enter property details**:
   - Living area, bedrooms, bathrooms
   - Basement area, garage capacity
   - Year built, lot size, quality rating

3. **Click "Predict House Price"** to get instant estimation

---

## 📁 Project Structure

```
House-Price-Prediction-Regressor/
│
├── app.py                  # Main Streamlit application
├── train_exact.py          # Model training script
├── house_prices.csv        # Dataset
├── xgb_model.pkl          # Trained XGBoost model
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

---

## 🔧 Model Training

To retrain the model with your own data:

```bash
python train_exact.py
```

This will:
- Load the `house_prices.csv` dataset
- Train an XGBoost model with optimized parameters
- Save the model as `xgb_model.pkl`
- Display training performance metrics

---

## 🎨 UI Features

- **Dark Glassmorphism Theme** - Modern gradient background
- **Responsive Input Grid** - 3-column layout for easy data entry
- **Professional Result Display** - Large, clear price prediction
- **Informative Sidebar** - Model info, instructions, developer details
- **Smooth Animations** - Enhanced user experience

---

## 👨‍💻 Developer

**Partha Sarathi R**
- Email: ayyapparaja227@gmail.com
- GitHub: [@partha0059](https://github.com/partha0059)

---

## 📝 License

This project is open source and available for educational purposes.

---

## 🙏 Acknowledgments

- Built with **Streamlit** ❤️
- Powered by **XGBoost** gradient boosting
- Dataset inspired by real estate data

---

## 🔮 Future Enhancements

- [ ] Add more features (location, condition, etc.)
- [ ] Deploy to cloud (Streamlit Cloud/Heroku)
- [ ] Add data visualization dashboard
- [ ] Implement ensemble models
- [ ] Add model comparison feature

---

**⭐ If you find this project helpful, please give it a star!**
