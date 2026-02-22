import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    hungry = request.form.get('hungry')
    weekend = request.form.get('weekend')
    
    # Convert to numeric (Yes=1, No=0)
    hungry_val = 1 if hungry == 'Yes' else 0
    weekend_val = 1 if weekend == 'Yes' else 0
    
    # Create numpy array for prediction
    input_data = np.array([[hungry_val, weekend_val]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Convert back to Yes/No
    result = "Yes, you should eat pizza! 🍕" if prediction == 1 else "No, maybe skip the pizza this time."
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
