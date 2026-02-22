import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
Fake News Detection - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import os

app = Flask(__name__)

# ============================================
# Load Model and Vectorizer
# ============================================
print("🔄 Loading model and vectorizer...")

try:
    with open('model/fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('model/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found! Please run 'python train_model.py' first.")
    model = None
    vectorizer = None
    metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

# ============================================
# Text Preprocessing Function
# ============================================
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================
# Routes
# ============================================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        # Get the news text from request
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text.strip():
            return jsonify({
                'success': False,
                'error': 'Please enter some news text to analyze.'
            })
        
        if model is None or vectorizer is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            })
        
        # Preprocess the text
        cleaned_text = preprocess_text(news_text)
        
        # Transform using TF-IDF vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Prepare result
        result = {
            'success': True,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'is_fake': bool(prediction == 1),
            'confidence': round(confidence, 2),
            'fake_probability': round(probability[1] * 100, 2),
            'real_probability': round(probability[0] * 100, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/metrics')
def get_metrics():
    """Return model metrics"""
    return jsonify({
        'accuracy': round(metrics['accuracy'] * 100, 2),
        'precision': round(metrics['precision'] * 100, 2),
        'recall': round(metrics['recall'] * 100, 2),
        'f1': round(metrics['f1'] * 100, 2)
    })

# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 FAKE NEWS DETECTION WEB APP")
    print("=" * 50)
    print("Open http://127.0.0.1:5050 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5050)
