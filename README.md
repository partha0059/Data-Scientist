# Data Scientist – Machine Learning Algorithms

This repository contains a collection of Machine Learning algorithms organized by Supervised, Unsupervised Learning and ML Engineering & Optimization, along with links to respective GitHub repositories for each algorithm.

---

## 📚 Learning Path

### Foundation
- What is AI? — High-level definitions of Artificial and Machine Learning
- Real-world ML Applications — Industry use cases and examples

### Mathematical Foundations
- Linear Algebra — Vectors, Matrices, Transformations
- Descriptive Statistics — Measures of central tendency and dispersion
- Probability Basics — Conditional Probability, Bayes Theorem
- NumPy Hands-on — Numerical computing fundamentals

### Python Ecosystem
- Python Libraries for ML — NumPy, Pandas, Matplotlib, Seaborn
- EDA, Preprocessing and Visualization — Data cleaning and exploration techniques

### Core ML Concepts
- What is a Model? — Target vs features. Overfitting vs underfitting

---

## 🤖 ML Algorithms

### Supervised Learning

#### Regression
- **Linear Regression** Theory & Hands-on | Evaluation Metrics | Project: Coffee Shop Sales Prediction via Streamlit Deployment -> [coffee-shop-sales](./Regression/Linear-Regression-Coffee-Shop-Sales-Regressor)
- **Lasso & Ridge Regression** Theory + Hands-on | Evaluation Metrics | Mini-Project: Lasso Linear Regression -> [Lasso_Linear](./Regression/Lasso-Linear-Regression)
- **Random Forest Regression** | Project: Energy Efficiency Prediction -> [Energy-Efficiency-Prediction](./Regression/RandomForest-Energy-Efficiency-Prediction-Regressor)
- **XGBoost Regression** | Project: House Price Prediction -> [House-Price-Prediction](./Regression/XGBoost-House-Price-Prediction-Regressor)
- **Gradient Boosting Regression** | Project: Diabetes Risk Prediction -> [Diabetes-Risk-Prediction](./Regression/Gradient-Boosting-Diabetes-Risk-Predictor-Regressor)
- **Cross-Validation Regression** | Project: Wine Quality Prediction -> [Wine-Quality-Prediction](./Regression/Cross-Validation-Wine-Quality-Prediction-Regressor)

#### Classification
- **Logistic Regression** Evaluation Metrics | Mini-Project: Heart Disease Classification -> [Heart-Disease-Prediction](./Classification/Logistic-Regression-Heart-Disease-Prediction)
- **K-Nearest Neighbors (KNN)** Theory + Hands-on | Mini-Project: Breast Cancer Classification -> [KNN-Breast-Cancer-Classification](./Classification/KNN-Breast-Cancer-Classification)
- **Naive Bayes Algorithm** Theory + Hands-on | Mini-Project: SMS Spam Classifier + Fake News Detector + Tennis Play Predictor -> [SMS-Spam-Prediction](./Classification/Multinomial-Naive-Bayes-SMS-Spam-Prediction-Classification) | [Fake-News-Detector](./Classification/Naive-Bayes-Fake-News-Detector-Classification) | [Tennis-Play-Predictor](./Classification/Naive-Bayes-Tennis-Play-Predictor-Classification)
- **Decision Trees** Gini, Entropy + Hands-on | Mini-Project: Movie Interest Predictor + Pizza Predictor -> [Movie-Interest-Predictor](./Classification/Decision-Tree-Movie-Interest-Predictor-Classifier) | [Pizza-Predictor](./Classification/Decision-Tree-Pizza-Predictor-Classifier)
- **Random Forest** Ensemble Learning | Project: Employee Performance Rating -> [Employee-Performance-Classifier](./Classification/RandomForest-Employee-Performance-Classifier)
- **Support Vector Machine (SVM)** Theory + Hands-on | Mini-Project: Digit Recognition -> [SVM-Digit-Predictor](./Classification/SVM-Digit-Predictor)
- **Gradient Boosting** Theory + Hands-on | Mini-Project: Mushroom Classification -> [Mushroom-Classification-Classifier](./Classification/Gradient-Boosting-Mushroom-Classification-Classifier)
- **XGBoost Algorithm** Theory + Hands-on | Project: Milk Quality Prediction -> [Milk-Quality-Classifier](./Classification/XGBoost-Milk-Quality-Classifier)

### Unsupervised Learning

#### Clustering
- **K-Means Clustering** Elbow Method, Inertia | Project: Customer Segmentation -> [K-Means-Customer-Segmentation](./Clustering/K-Means-Customer-Segmentation)
- **Hierarchical Clustering** -> [Hierarchical-Clustering](./Clustering/Hierarchical-Clustering)
- **DBSCAN** Density-based clustering | Mini-Project: Wine Clustering Analysis -> [DBSCAN-Wine-Clustering](./Clustering/DBSCAN-Wine-Clustering)

#### Dimensionality Reduction
- **PCA (Principal Component Analysis)** Real-time Data application | Feature Extraction (*No specific project currently*)

---

## ⚙️ ML Engineering & Optimization

### Model Optimization
- Bias/Variance Tradeoff - Overfitting/Underfitting analysis

### Cross Validation
- Cross Validation (K-Fold) - Prevent Overfitting, Verify Generalization -> [Wine-Quality-Prediction-Regressor](./Regression/Cross-Validation-Wine-Quality-Prediction-Regressor)

### Hyperparameter Tuning
- GridSearch CV, RandomizedSearch CV - Best Parameter for ML Models

### Feature Selection
- Feature Methods - Filter, Wrapper Methods & Embedded Methods -> [Feature-Selection-Filter-Wrapper](./Feature-Selection/Feature-Selection-Filter-Wrapper)

---

## 🚀 Projects Portfolio

| Project | Algorithm | Tech Stack | Repository |
|---|---|---|---|
| Coffee Shop Sales Prediction | Linear Regression | Python, Streamlit | [coffee-shop-sales](./Regression/Linear-Regression-Coffee-Shop-Sales-Regressor) |
| House Price Prediction | XGBoost | Python, Streamlit | [House-Price-Prediction-Regressor](./Regression/XGBoost-House-Price-Prediction-Regressor) |
| Heart Disease Prediction | Logistic Regression | Python, Flask | [Heart-Disease-Prediction](./Classification/Logistic-Regression-Heart-Disease-Prediction) |
| Breast Cancer Classification | KNN | Python, Streamlit | [KNN-Breast-Cancer-Classification](./Classification/KNN-Breast-Cancer-Classification) |
| Milk Quality Classifier | XGBoost | Python, Streamlit | [Milk-Quality-Classifier](./Classification/XGBoost-Milk-Quality-Classifier) |
| SMS Spam Prediction | Naive Bayes | Python, Streamlit | [SMS-Spam-Prediction](./Classification/Multinomial-Naive-Bayes-SMS-Spam-Prediction-Classification) |
| Fake News Detector | Naive Bayes / NLP | Python, Flask + Render | [Fake-News-Detector](./Classification/Naive-Bayes-Fake-News-Detector-Classification) |
| Pizza Predictor | Decision Tree | Python, Flask | [Pizza-Predictor-Decision-Tree](./Classification/Decision-Tree-Pizza-Predictor-Classifier) |
| Tennis Play Predictor | Decision Tree | Python, Streamlit | [Tennis-Play-Predictor](./Classification/Naive-Bayes-Tennis-Play-Predictor-Classification) |
| Employee Performance Rating | Random Forest | Python, Streamlit | [RandomForest-employee-performance](./Classification/RandomForest-Employee-Performance-Classifier) |
| Energy Efficiency Prediction | Random Forest | Python, Streamlit | [RandomForest-Energy-Efficiency-Prediction-Regressor](./Regression/RandomForest-Energy-Efficiency-Prediction-Regressor) |
| SVM Digit Predictor | Support Vector Machine | Python, Jupyter | [SVM-Digit-Predictor](./Classification/SVM-Digit-Predictor) |
| Mushroom Classification | Gradient Boosting | Python, Streamlit | [Mushroom-Classification-System](./Classification/Gradient-Boosting-Mushroom-Classification-Classifier) |
| Diabetes Prediction | Gradient Boosting | Python, Streamlit | [Diabetes-Gradient-Boosting](./Regression/Gradient-Boosting-Diabetes-Risk-Predictor-Regressor) |
| Wine Quality Prediction | Cross Validation | Python, Streamlit | [Wine-Quality-Prediction-Regressor](./Regression/Cross-Validation-Wine-Quality-Prediction-Regressor) |
| Wine Clustering | DBSCAN | Python, Streamlit | [DBSCAN-Wine-Clustering](./Clustering/DBSCAN-Wine-Clustering) |
| Customer Segmentation | K-Means Clustering | Python, Streamlit | [K-mean-Customer-Segmentation](./Clustering/K-Means-Customer-Segmentation) |
| Hierarchical Clustering | Hierarchical Clustering | Python, Streamlit | [Hierarchical-Clustering](./Clustering/Hierarchical-Clustering) |
| Movie Interest Predictor | Decision Tree | Python, Flask | [Movie-Interest-Predictor](./Classification/Decision-Tree-Movie-Interest-Predictor-Classifier) |
| Lasso Linear Regression | Lasso / Ridge | Python, Jupyter | [Lasso_Linear](./Regression/Lasso-Linear-Regression) |
| Feature Selection | Filter & Wrapper | Python, Streamlit | [Feature-Selection-Filter-Wrapper](./Feature-Selection/Feature-Selection-Filter-Wrapper) |

---

## 👨‍💻 About

Connect to a network of machine learning specialists, data programming framework, broad datasets, and build modern ML projects — Discovering new concepts, required CV & AI domain related roles.

Developer: Partha Sarathi R  
GitHub: [partha0059](https://github.com/partha0059)
