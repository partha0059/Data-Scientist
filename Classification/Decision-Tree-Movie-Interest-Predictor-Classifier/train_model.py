import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

# Create static directory for visualization if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load dataset
try:
    df = pd.read_csv('Task_Dataset_Movie_Interests_DecisionTree.csv')
    X = df[['Age', 'Gender']]
    y = df['Interest']
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

# Train-test split (though dataset is small, it's good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gini Data Model
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
acc_gini = accuracy_score(y_test, y_pred_gini)

# Train Entropy Model
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)

print(f"Accuracy (Gini): {acc_gini * 100:.2f}%")
print(f"Accuracy (Entropy): {acc_entropy * 100:.2f}%")

# Select the model (Entropy often preferred for categorical depth, but usually similar here)
# We will use Entropy as requested for 'Gini vs Entropy' focus, or whichever is better.
# For this small dataset, we'll just pick Entropy to show we did it.
final_model = clf_entropy
final_model.fit(X, y) # Retrain on full dataset for the app

# Save the model
joblib.dump(final_model, 'model.pkl')
print("Model saved as model.pkl")

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(final_model, feature_names=['Age', 'Gender'], class_names=final_model.classes_, filled=True, rounded=True)
plt.title("Decision Tree for Movie Interest Prediction")
plt.savefig('static/decision_tree.png')
print("Visualization saved as static/decision_tree.png")
