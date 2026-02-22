import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import json

def train_and_save_model():
    """Train Linear Regression model with K-Fold Cross-Validation on Wine Quality dataset"""
    
    print("Loading Wine Quality dataset...")
    
    # Try loading from local file first, then URL
    try:
        data = pd.read_csv('winequality-red.csv', sep=';')
        print("Loaded from local file")
    except FileNotFoundError:
        print("Downloading from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url, sep=';')
    
    print(f"Dataset loaded: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    
    # Prepare data
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Initialize model
    model = LinearRegression()
    
    # K-Fold Cross-Validation (5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scoring function
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    print("\nPerforming K-Fold Cross-Validation...")
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    rmse_scores = np.sqrt(-cv_scores)
    
    print("\nCross-Validation Results:")
    print(f"RMSE for each fold: {rmse_scores}")
    print(f"Average RMSE: {rmse_scores.mean():.4f}")
    print(f"Standard Deviation: {rmse_scores.std():.4f}")
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    model.fit(X, y)
    
    # Save model
    print("\nSaving model and results...")
    joblib.dump(model, 'wine_quality_model.pkl')
    
    # Save cross-validation results
    cv_results = {
        'rmse_scores': rmse_scores.tolist(),
        'mean_rmse': float(rmse_scores.mean()),
        'std_rmse': float(rmse_scores.std()),
        'n_splits': 5,
        'feature_names': X.columns.tolist(),
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_)
    }
    
    with open('cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print("\n✓ Model and results saved successfully!")
    print("  - wine_quality_model.pkl")
    print("  - cv_results.json")

if __name__ == "__main__":
    train_and_save_model()
