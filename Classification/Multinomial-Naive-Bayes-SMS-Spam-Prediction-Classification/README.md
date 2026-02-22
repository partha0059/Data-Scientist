# SMS Intelligence System 🛡️

An advanced Natural Language Processing (NLP) system designed for real-time threat detection and classification of SMS messages. This project utilizes machine learning to distinguish between legitimate (Ham) and fraudulent (Spam) messages with extremely high precision.

## 🚀 Live Demo
The application is optimized for Streamlit deployment. You can run it locally or deploy it to Streamlit Cloud.

---

## 📊 Technical Analysis

### Model Performance
- **Algorithm**: Multinomial Naive Bayes (MNB)
- **Vectorization**: CountVectorizer
- **Accuracy**: **99%**
- **Precision (Spam)**: **99%**
- **Recall (Spam)**: **92%**
- **Dataset**: 5,572 verified message patterns

### Why Multinomial Naive Bayes?
The system uses the Multinomial Naive Bayes algorithm, which is highly effective for text classification tasks where features represent the frequencies with which certain words occur. It applies Bayes' theorem with strong independence assumptions between features.

---

## ✨ Key Features
- **Real-time Analysis**: Instant classification with < 0.05s latency.
- **Interactive Dashboard**: Professional white & blue theme with glassmorphism effects.
- **Threat Monitoring**: Global statistics and distribution of the training dataset.
- **Detailed Reporting**: confidence metrics and risk assessment levels (CRITICAL, HIGH, LOW).
- **Export Capabilities**: Download analysis reports as text or JSON.

---

## 🛠️ Project Structure
```text
SMS_Spam_Project/
├── app.py              # Streamlit Application
├── SMS_Spam.ipynb       # EDA & Model Training Notebook
├── sms_spam.pkl         # Trained Model Binary
├── vectorizer.pkl       # Vectorizer Binary
├── spam.csv             # Training Dataset
└── requirements.txt     # Python Dependencies
```

---

## 💻 Installation & Usage

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/partha0059/SMS-Spam-Prediction.git
   cd SMS_Spam_Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 👨‍💻 Developer
Developed and Engineered by **Partha Sarathi R**
Focus: Data Science & AI Research
