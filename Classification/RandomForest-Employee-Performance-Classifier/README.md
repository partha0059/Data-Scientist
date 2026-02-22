# 📊 Employee Performance Rating Predictor

A professional web application built with Streamlit that predicts employee performance ratings using a Random Forest machine learning model.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **AI-Powered Predictions**: Uses Random Forest classifier with 100 decision trees
- **Modern UI**: Dark glassmorphism design with smooth animations
- **Interactive Visualizations**: 
  - Confidence gauge charts
  - Probability distribution graphs
  - Feature importance analysis
- **Real-time Analytics**: Instant performance predictions with detailed insights
- **Comprehensive Reporting**: Personalized recommendations based on predictions
- **Multi-tab Interface**: Organized sections for predictions, analytics, and information

## 🎯 Prediction Factors

The model analyzes 7 key employee attributes:

1. **Age** - Employee's current age
2. **Experience Years** - Total professional experience
3. **Department** - Work department (IT, HR, Sales, Finance)
4. **Salary** - Annual compensation
5. **Work Hours** - Weekly working hours
6. **Projects Handled** - Current project load
7. **Training Hours** - Monthly training participation

## 📋 Performance Ratings

- **0 - Needs Improvement**: Requires additional support and development
- **1 - Excellent Performance**: Exceeds expectations, ready for advancement

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
   ```bash
   cd TestDataset
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```
   
   This will create:
   - `random_forest_model.pkl` - Trained Random Forest model
   - `label_encoder.pkl` - Department label encoder

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
TestDataset/
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── random_forest_model.pkl         # Trained model (generated)
├── label_encoder.pkl              # Label encoder (generated)
└── Randomforest.ipynb             # Original notebook
```

## 🎨 User Interface

### Predict Performance Tab
- Input employee details through an intuitive form
- Click "Predict Performance Rating" for instant results
- View confidence scores with interactive gauge charts
- Get personalized recommendations

### Model Analytics Tab
- Explore model performance metrics
- Visualize feature importance
- Understand which factors drive predictions

### Information Tab
- Learn about the Random Forest algorithm
- Understand model features and technical details
- Explore use cases and applications

## 🔧 Technical Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Max Depth**: 10 levels
- **Criterion**: Gini impurity
- **Random State**: 42 (for reproducibility)
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts

## 💡 Use Cases

- **HR Departments**: Evaluate employee performance objectively
- **Talent Management**: Identify high-performers and areas needing support
- **Training Programs**: Assess training effectiveness and needs
- **Career Development**: Plan employee growth paths
- **Resource Allocation**: Optimize team compositions

## 📊 Model Performance

- **Accuracy**: 100% on test set
- **Features**: 7 employee attributes
- **Training Data**: 20 employee records (expandable)

## 🎓 About the Developer

**Created by**: Partha Sarathi R

This project demonstrates the application of machine learning in HR analytics, combining modern web development with AI-powered predictions.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please feel free to reach out.

---

<div align="center">
  <p>Developed with ❤️ using Streamlit & Scikit-learn</p>
  <p>© 2026 Employee Performance Analytics</p>
</div>
