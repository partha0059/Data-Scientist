import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# Prepare dataset
data = {
    'Am_I_Hungry': ['Yes', 'Yes', 'Yes', 'No', 'No'],
    'Is_It_Weekend': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Shall_I_Eat_Pizza': ['Yes', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)

# Convert Yes/No to 1/0
# Note: In the notebook, replacement was direct. We will use the same logic for consistency.
# Hungry: Yes=1, No=0
# Weekend: Yes=1, No=0
df_encoded = df.replace({'Yes': 1, 'No': 0})

# Features and target
X = df_encoded[['Am_I_Hungry', 'Is_It_Weekend']]
y = df_encoded['Shall_I_Eat_Pizza']

# Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=3)
model.fit(X, y)

# Save the model
model_path = 'decision_tree_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {os.path.abspath(model_path)}")
print("Classes:", model.classes_)

# Verify predictions (sanity check)
test_data = pd.DataFrame([[0, 1], [1, 0]], columns=['Am_I_Hungry', 'Is_It_Weekend']) # No, Yes | Yes, No
print("Predictions for test data [[0, 1], [1, 0]]:", model.predict(test_data))
