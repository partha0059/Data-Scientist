#!/bin/bash
set -o errexit

echo "🚀 Starting Build Process..."

# 1. Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. Train the model (Required because model files are not in Git)
echo "🧠 Training model and generating vectorizer..."
python train_model.py

echo "✅ Build script completed successfully!"
