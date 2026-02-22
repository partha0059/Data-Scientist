"""
Employee Performance Rating Model Training Script
This script trains a Random Forest classifier and saves the model and encoder
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    """Train Random Forest model and save artifacts"""
    
    # Sample dataset (in production, load from CSV)
    data = {
        "Age": [28, 35, 30, 42, 25, 38, 29, 45, 33, 27, 40, 31, 26, 37, 43, 32, 36, 28, 41, 34],
        "Experience_Years": [4, 10, 6, 15, 2, 12, 5, 18, 8, 3, 14, 7, 1, 11, 16, 9, 13, 4, 15, 10],
        "Department": ["IT", "HR", "Sales", "Finance", "IT", "Sales", "HR", "Finance", "IT", "Sales", 
                      "Finance", "HR", "IT", "Sales", "Finance", "HR", "IT", "Sales", "Finance", "HR"],
        "Salary": [45000, 60000, 50000, 75000, 40000, 65000, 48000, 80000, 55000, 42000, 
                  72000, 52000, 38000, 62000, 78000, 54000, 68000, 46000, 74000, 58000],
        "Work_Hours": [42, 40, 45, 38, 44, 41, 43, 39, 42, 46, 37, 44, 47, 40, 38, 43, 39, 45, 37, 41],
        "Projects_Handled": [3, 5, 4, 6, 2, 5, 3, 7, 4, 2, 6, 4, 1, 5, 7, 4, 6, 3, 6, 5],
        "Training_Hours": [20, 15, 25, 10, 30, 18, 22, 8, 16, 28, 12, 24, 32, 14, 9, 19, 11, 26, 10, 17],
        "Performance_Rating": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    
    print("Dataset Overview:")
    print(df.head())
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nPerformance Rating Distribution:\n{df['Performance_Rating'].value_counts()}")
    
    # Encode Department
    le = LabelEncoder()
    df["Department"] = le.fit_transform(df["Department"])
    
    # Split features and target
    X = df.drop("Performance_Rating", axis=1)
    y = df["Performance_Rating"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=42,
        max_depth=10,
        min_samples_split=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Model Training Complete!")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:\n{feature_importance}")
    
    # Save model and encoder
    joblib.dump(model, "random_forest_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    
    print(f"\n{'='*60}")
    print("✓ Model saved as 'random_forest_model.pkl'")
    print("✓ Label Encoder saved as 'label_encoder.pkl'")
    print(f"{'='*60}")
    
    return model, le

if __name__ == "__main__":
    train_and_save_model()
