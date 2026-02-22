# 🍷 Wine Quality Prediction - K-Fold Cross-Validation

A professional Streamlit web application for predicting wine quality using Linear Regression with K-Fold Cross-Validation. This interactive app allows users to predict wine quality based on physicochemical properties and explore model performance through comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 🔮 Wine Quality Prediction
- **Interactive Input Form**: Enter 11 physicochemical wine properties
- **Instant Predictions**: Get quality scores (0-10 scale) in real-time
- **Quality Categories**: Automatic classification (Low, Medium, Good, Excellent)
- **Comparison Analysis**: View your inputs vs. dataset averages

### 📊 K-Fold Cross-Validation
- **Adjustable Folds**: Configure between 3-20 folds
- **Performance Metrics**: Mean RMSE, Standard Deviation, Best Fold
- **Interactive Charts**: 
  - RMSE per fold bar chart
  - Score distribution box plot
- **Detailed Results Table**: Fold-by-fold breakdown

### 📈 Model Analysis
- **Feature Coefficients Chart**: Visualize feature importance
- **Dataset Statistics**: Comprehensive statistical summary
- **Model Parameters**: View intercept and coefficients

### 🎨 Professional UI/UX
- Clean white background with excellent contrast
- Dark, readable text throughout
- Interactive visualizations with Plotly
- Responsive design with smooth animations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/partha0059/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
Wine-Quality-Prediction/
├── app.py                    # Main Streamlit application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 📊 Dataset

This project uses the **UCI Wine Quality Dataset** (Red Wine variant):
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples**: 1,599 red wine samples
- **Features**: 11 physicochemical properties
- **Target**: Wine quality rating (0-10 scale)

### Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| Fixed Acidity | Non-volatile acids | g/dm³ |
| Volatile Acidity | Acetic acid amount | g/dm³ |
| Citric Acid | Adds freshness and flavor | g/dm³ |
| Residual Sugar | Remaining sugar after fermentation | g/dm³ |
| Chlorides | Salt amount | g/dm³ |
| Free Sulfur Dioxide | Free form of SO₂ | mg/dm³ |
| Total Sulfur Dioxide | Total SO₂ (free + bound) | mg/dm³ |
| Density | Wine density | g/cm³ |
| pH | Acidity level | - |
| Sulphates | Wine additive | g/dm³ |
| Alcohol | Alcohol percentage | % vol |

## 🤖 Model Details

### Algorithm
- **Type**: Linear Regression
- **Validation**: K-Fold Cross-Validation
- **Default Folds**: 5
- **Random State**: 42 (reproducible results)

### Performance
- **Mean RMSE**: ~0.6536
- **Standard Deviation**: ~0.0400
- **Evaluation**: Consistent performance across folds

## 🎯 Usage Examples

### Predicting Wine Quality

1. Navigate to the **🔮 Predict Quality** tab
2. Enter wine properties (or use default values)
3. Click **"Predict Wine Quality"**
4. View the predicted quality score and category

### Running Cross-Validation

1. Go to the **📈 Cross-Validation** tab
2. Adjust the number of folds (3-20)
3. Set random state for reproducibility
4. Click **"Run Cross-Validation"**
5. Analyze RMSE metrics and charts

### Analyzing Features

1. Visit the **📊 Model Analysis** tab
2. Explore feature coefficients chart
3. Review dataset statistics
4. Check model parameters

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Model Persistence**: Joblib

## 📈 Application Screenshots

### Predict Quality Tab
Predict wine quality by entering physicochemical properties and get instant results with quality categorization.

### Cross-Validation Tab
Explore model performance with interactive charts showing RMSE across different folds.

### Model Analysis Tab
Visualize feature importance and understand which factors most influence wine quality predictions.

## 🔧 Model Training

The model is automatically trained when you run `train_model.py`. The script:

1. Downloads the Wine Quality dataset from UCI repository
2. Performs 5-fold cross-validation
3. Trains Linear Regression model on full dataset
4. Saves model as `wine_quality_model.pkl`
5. Saves validation results as `cv_results.json`

## 📝 Requirements

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
plotly==5.18.0
joblib==1.3.2
```

## 🎓 Educational Value

This project demonstrates:
- K-Fold Cross-Validation implementation
- Linear Regression for regression tasks
- Interactive data visualization
- Professional Streamlit app development
- Model evaluation best practices

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Developer

**Partha Sarathi R**

- GitHub: [@partha0059](https://github.com/partha0059)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- Streamlit team for the amazing framework
- scikit-learn developers for comprehensive ML tools

## 📞 Support

If you encounter any issues or have questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and screenshots

---

⭐ If you find this project helpful, please consider giving it a star!
