# 🕵️ Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated **Machine Learning** application designed to detect fake news with high accuracy (**96.2%**). This system utilizes Natural Language Processing (NLP) techniques to analyze text patterns and classify news articles as **REAL** or **FAKE**.

It combines traditional datasets (ISOT) with a custom-generated **Modern 2024-2025 Dataset** to ensure detection capabilities extend to current misinformation trends, including AI-generated deepfakes, election fraud claims, and modern health scams.

---

## 📊 Project Overview

In the age of digital information, misinformation spreads rapidly. This project aims to build a robust tool that can:
1.  **Analyze** news articles in real-time.
2.  **Classify** content as reliable or deceptive.
3.  **Provide Confidence Scores** to help users make informed decisions.
4.  **Visualize** common patterns in fake vs. real news.

### 🌟 Key Features
*   **Hybrid Dataset approach**: Combines historical data (2016-2017) with modern synthetic data (2024-2025).
*   **Advanced NLP**: Uses TF-IDF Vectorization with n-grams (1-3) to capture context, not just keywords.
*   **High Performance**: Achieves >96% accuracy using a tuned Multinomial Naive Bayes classifier.
*   **Web Interface**: Clean, responsive Flask interface for easy user interaction.
*   **Data Visualization**: Real-time generation of WordClouds and Confidence Metrics.

---

## 💾 Dataset Analysis

The model is trained on a massive combined dataset of over **45,000 articles**, ensuring broad coverage of various writing styles and topics.

| Dataset Source | Description | Size |
| :--- | :--- | :--- |
| **ISOT Fake News** | Articles containing deliberate misinformation (2016-2017). | ~23,481 |
| **ISOT Real News** | Verified articles from Reuters (2016-2017). | ~21,417 |
| **Modern 2024-2025** | Custom dataset representing current trends (AI, Deepfakes, 2024 Elections, Health Scams). | ~500 |
| **Total** | **Combined & Balanced** | **~45,000+** |

### 🧠 Data Processing Pipeline
1.  **Cleaning**:
    *   Removal of URLs, HTML tags, and special characters.
    *   Handling of "Reuters" tags in real news to prevent bias.
    *   Conversion to lowercase for uniformity.
2.  **Feature Engineering**:
    *   **Text Preprocessing**: Custom logic to retain "excessive capitalization" and "excessive punctuation" (!!!) as these are strong indicators of fake news.
    *   **Vectorization**: `TfidfVectorizer` with **15,000 features** and **bi-grams/tri-grams** to capture phrases like "shocking revelation" or "official source".

---

## ⚙️ Methodology & Architecture

The project follows a standard Machine Learning lifecycle:

### 1. Model Selection
We chose **Multinomial Naive Bayes (MNB)** because:
*   It excels at text classification tasks.
*   It handles high-dimensional data (15,000+ features) efficiently.
*   It is computationally fast for real-time web predictions.
*   It provides probability estimates (Confidence Scores) naturally.

### 2. Training Process (`train_model.py`)
*   **Split**: 80% Training / 20% Testing.
*   **Validation**: Stratified sampling to ensure balanced classes.
*   **Hyperparameter Tuning**: Alpha Smoothing (alpha=0.05) optimized for best F1-Score.

### 3. Application Architecture (`app.py`)
*   **Frontend**: HTML5/CSS3 with a modern, clean design.
*   **Backend**: Flask (Python) server.
*   **Inference**: Loads the pre-trained `fake_news_model.pkl` and `tfidf_vectorizer.pkl` into memory for sub-second predictions.

---

## 📈 Performance & Results

The model has been rigorously evaluated on a held-out test set of **~9,000 articles**.

| Metric | Score | Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **96.22%** | Correctly classifies 96 out of 100 articles. |
| **Precision** | **96.28%** | When it says "Fake", it's almost always right. |
| **Recall** | **96.52%** | It catches 96.5% of all fake news in the dataset. |
| **F1-Score** | **96.40%** | Perfect balance between Precision and Recall. |

> **Note**: The model is particularly effective at detecting "Clickbait" style headlines and emotionally charged language standard in misinformation.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/partha0059/Fake-News-Detector.git
cd Fake-News-Detector
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (Required)
**Why?** The vectorizer file (`tfidf_vectorizer.pkl`) is ~200MB and cannot be hosted on GitHub. You **must** generate it locally.
```bash
python train_model.py
```
*This script will:*
1. *Combine datasets.*
2. *Train the Naive Bayes model.*
3. *Generate the `model/` files and `static/images/` visualizations.*

### Step 4: Run the Application
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5050`

---

## 🔮 Future Scope
*   **Deep Learning**: Implement LSTM or BERT models for better context understanding (at the cost of speed).
*   **URL Analysis**: Add feature to scrape and analyze content directly from a URL.
*   **Chrome Extension**: Build a browser plugin to flag news as you browse social media.
*   **Multilingual Support**: Expand dataset to include news in Spanish, French, and Hindi.

---

## 👨‍💻 Author
**Partha Sarathi R**
*   **GitHub**: [partha0059](https://github.com/partha0059)
