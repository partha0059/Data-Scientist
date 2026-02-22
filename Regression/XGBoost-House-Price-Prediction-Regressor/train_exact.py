import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("="*70)
print("TRAINING FOR EXACT PREDICTIONS (MAXIMUM OVERFITTING)")
print("="*70)

# Load dataset
df = pd.read_csv("house_prices.csv")

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'TotalBsmtSF', 
            'GarageCars', 'YearBuilt', 'LotArea', 'OverallQual']

X = df[features]
y = df['SalePrice']

print(f"\nDataset: {len(df)} houses")

# Split with same random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# Train with parameters that MAXIMIZE overfitting (memorization)
print("\nTraining with MAXIMUM memorization capability...")

model = XGBRegressor(
    n_estimators=5000,      # Many trees
    learning_rate=0.01,     # Slow learning
    max_depth=15,           # Very deep trees (memorize details)
    min_child_weight=1,     
    subsample=1.0,          # Use all data (no sampling)
    colsample_bytree=1.0,   # Use all features
    gamma=0,                # No pruning
    reg_alpha=0,            # NO regularization
    reg_lambda=0,           # NO regularization
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, verbose=False)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n{'='*70}")
print("PERFORMANCE")
print("="*70)
print(f"Training R²: {train_r2:.6f} ({train_r2*100:.2f}%) - Training MAE: ${train_mae:,.2f}")
print(f"Test R²:     {test_r2:.6f} ({test_r2*100:.2f}%) - Test MAE: ${test_mae:,.2f}")

# Test on ALL houses in dataset
print(f"\n{'='*70}")
print("PREDICTIONS ON FULL DATASET (First 10 houses)")
print("="*70)

for i in range(min(10, len(X))):
    actual = y.iloc[i]
    predicted = model.predict(X.iloc[i:i+1])[0]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100
    
    print(f"\nHouse {i+1}:")
    print(f"  Actual:    ${actual:>10,.0f}")
    print(f"  Predicted: ${predicted:>10,.0f}")
    print(f"  Error:     ${error:>10,.0f} ({error_pct:.2f}%)")

# Save model
joblib.dump(model, "xgb_model.pkl")
print(f"\n✅ Model saved as 'xgb_model.pkl'")

print("\n" + "="*70)
print("NOTE: This model is OVERFITTED to memorize training data.")
print("It will give very close predictions for houses in the dataset,")
print("but may not generalize well to completely new houses.")
print("="*70)
