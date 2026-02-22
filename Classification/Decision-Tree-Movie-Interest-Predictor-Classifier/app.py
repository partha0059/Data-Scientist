import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Please train the model first.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            
            # Create a DataFrame for prediction to match training data structure
            input_data = pd.DataFrame([[age, gender]], columns=['Age', 'Gender'])
            
            if model:
                prediction = model.predict(input_data)[0]
        except ValueError:
            prediction = "Invalid Input"
        except Exception as e:
            prediction = f"Error: {e}"
            
    return render_template('index.html', prediction=prediction)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    except ValueError:
        # Fallback for environments where signal handlers cannot be set (e.g. non-main threads)
        print("Warning: Signal handling failed. Running without reloader.")
        app.run(debug=True, use_reloader=False, port=5001)
