"""
Fake News Detection - Combined Dataset Training Script
Combines ISOT Dataset (2016-2017) + Modern 2024-2025 Patterns
For comprehensive fake news detection
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('data/recent', exist_ok=True)

print("=" * 70)
print("🔍 FAKE NEWS DETECTION - COMBINED DATASET TRAINING")
print("   ISOT Dataset (44,000+) + Modern 2024-2025 Patterns")
print("=" * 70)

# ============================================
# STEP 1: Generate Modern 2024 Dataset
# ============================================
print("\n📂 Step 1: Generating Modern 2024-2025 Fake News Patterns...")
exec(open('generate_modern_dataset.py').read())

# ============================================
# STEP 2: Load All Datasets
# ============================================
print("\n📂 Step 2: Loading all datasets...")

# Load ISOT Dataset
print("   Loading ISOT True.csv (Real news from Reuters)...")
df_real = pd.read_csv('data/True.csv')
df_real['label'] = 0
df_real['source'] = 'ISOT'

print("   Loading ISOT Fake.csv (Fake news articles)...")
df_fake = pd.read_csv('data/Fake.csv')
df_fake['label'] = 1
df_fake['source'] = 'ISOT'

# Load Modern 2024 Dataset
print("   Loading Modern 2024-2025 patterns...")
df_modern = pd.read_csv('data/recent/modern_fake_news_2024.csv')
df_modern['source'] = 'Modern2024'

# Combine all datasets
df_isot = pd.concat([df_real, df_fake], ignore_index=True)
df_combined = pd.concat([df_isot, df_modern], ignore_index=True)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Combined Dataset loaded successfully!")
print(f"   📊 Total articles: {len(df_combined):,}")
print(f"   ├── ISOT Real news: {len(df_real):,}")
print(f"   ├── ISOT Fake news: {len(df_fake):,}")
print(f"   └── Modern 2024-2025: {len(df_modern):,}")
print(f"   📊 Total Real: {len(df_combined[df_combined['label']==0]):,}")
print(f"   📊 Total Fake: {len(df_combined[df_combined['label']==1]):,}")

# ============================================
# STEP 3: Enhanced Text Preprocessing
# ============================================
print("\n🔧 Step 3: Preprocessing text data...")

def preprocess_text(text):
    """Enhanced text preprocessing for modern fake news detection"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove Reuters tag
    text = re.sub(r'\(reuters\)', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep exclamation marks as features (common in fake news)
    # Count exclamations before removal
    exclaim_count = text.count('!')
    if exclaim_count > 3:
        text = text + ' excessive_exclamation ' * (exclaim_count // 3)
    
    # Check for all caps words (common in fake news)
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', str(text)))
    if caps_words > 2:
        text = text + ' excessive_caps ' * (caps_words // 2)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Combine title and text
print("   Combining title and text...")
df_combined['combined_text'] = df_combined['title'].fillna('') + ' ' + df_combined['text'].fillna('')

print("   Cleaning text data...")
df_combined['clean_text'] = df_combined['combined_text'].apply(preprocess_text)

# Remove short rows
df_combined = df_combined[df_combined['clean_text'].str.len() > 50]

print(f"✅ Text preprocessing complete!")
print(f"   📊 Final dataset size: {len(df_combined):,} articles")

# ============================================
# STEP 4: Feature Extraction using TF-IDF
# ============================================
print("\n📊 Step 4: Extracting features with TF-IDF...")

X = df_combined['clean_text']
y = df_combined['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   📈 Training samples: {len(X_train):,}")
print(f"   📈 Testing samples: {len(X_test):,}")

# Enhanced TF-IDF with more features for modern patterns
print("   Creating TF-IDF vectors...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,          # More features for combined dataset
    stop_words='english',
    ngram_range=(1, 3),          # Include trigrams for better patterns
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ TF-IDF vectorization complete!")
print(f"   📊 Features extracted: {X_train_tfidf.shape[1]:,}")

# ============================================
# STEP 5: Train Naive Bayes Model
# ============================================
print("\n🧠 Step 5: Training Multinomial Naive Bayes classifier...")

model = MultinomialNB(alpha=0.05)  # Lower alpha for better fit
model.fit(X_train_tfidf, y_train)

print("✅ Model training complete!")

# ============================================
# STEP 6: Evaluate Model
# ============================================
print("\n📈 Step 6: Evaluating model performance...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*60}")
print("📊 MODEL PERFORMANCE - COMBINED DATASET")
print(f"{'='*60}")
print(f"   🎯 Accuracy:  {accuracy*100:.2f}%")
print(f"   🎯 Precision: {precision*100:.2f}%")
print(f"   🎯 Recall:    {recall*100:.2f}%")
print(f"   🎯 F1-Score:  {f1*100:.2f}%")
print(f"{'='*60}")

if accuracy >= 0.95:
    print("   ✅ EXCELLENT: 95%+ Accuracy achieved!")
elif accuracy >= 0.90:
    print("   ✅ GOOD: 90%+ Accuracy achieved!")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

# ============================================
# STEP 7: Generate Visualizations
# ============================================
print("\n🎨 Step 7: Generating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Confusion Matrix
print("   - Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'],
            annot_kws={'size': 20})
plt.title(f'Confusion Matrix - Combined Dataset\nISOT + 2024 Modern Patterns | Accuracy: {accuracy*100:.1f}%', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Accuracy Chart
print("   - Creating accuracy chart...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']

plt.figure(figsize=(12, 7))
bars = plt.bar(metrics_names, values, color=colors, edgecolor='white', linewidth=3)
plt.ylim(0, 105)
plt.ylabel('Score (%)', fontsize=14, fontweight='bold')
plt.title(f'Model Performance - Combined Dataset\nISOT (44K) + Modern 2024 Patterns | Naive Bayes', 
          fontsize=16, fontweight='bold', pad=20)

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')

plt.axhline(y=95, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 95%')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('static/images/accuracy_chart.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Word Cloud - Fake News
print("   - Creating word cloud for fake news...")
fake_text = ' '.join(df_combined[df_combined['label'] == 1]['clean_text'].sample(min(5000, len(df_combined[df_combined['label']==1]))).tolist())
wordcloud_fake = WordCloud(
    width=1000, height=500,
    background_color='white',
    colormap='Reds',
    max_words=150,
    contour_width=2,
    contour_color='#ff4444'
).generate(fake_text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - FAKE NEWS\nCommon patterns in misinformation (2016-2025)', 
          fontsize=18, fontweight='bold', pad=20, color='#ff4444')
plt.tight_layout()
plt.savefig('static/images/wordcloud_fake.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Word Cloud - Real News
print("   - Creating word cloud for real news...")
real_text = ' '.join(df_combined[df_combined['label'] == 0]['clean_text'].sample(min(5000, len(df_combined[df_combined['label']==0]))).tolist())
wordcloud_real = WordCloud(
    width=1000, height=500,
    background_color='white',
    colormap='Greens',
    max_words=150,
    contour_width=2,
    contour_color='#00cc66'
).generate(real_text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - REAL NEWS\nCommon patterns in authentic journalism', 
          fontsize=18, fontweight='bold', pad=20, color='#00cc66')
plt.tight_layout()
plt.savefig('static/images/wordcloud_real.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("✅ All visualizations saved!")

# ============================================
# STEP 8: Save Model and Vectorizer
# ============================================
print("\n💾 Step 8: Saving model and vectorizer...")

with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

metrics_data = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'total_articles': len(df_combined),
    'dataset': 'ISOT + Modern 2024-2025',
    'features': X_train_tfidf.shape[1]
}
with open('model/metrics.pkl', 'wb') as f:
    pickle.dump(metrics_data, f)

print("✅ All files saved!")

# ============================================
# STEP 9: Test with Modern Samples
# ============================================
print("\n🧪 Step 9: Testing with modern fake news samples...")

test_samples = [
    # Modern AI fake news
    ("SHOCKING: ChatGPT is secretly recording all your conversations and selling them to the government!", "FAKE"),
    # Legitimate AI news
    ("OpenAI announced updates to GPT-4 with improved safety measures, according to the company's blog post.", "REAL"),
    # Modern deepfake fake news
    ("BREAKING: AI clones of celebrities are committing crimes! The real stars are being framed!", "FAKE"),
    # Legitimate tech news
    ("Google researchers published findings on improved machine learning efficiency in Nature journal.", "REAL"),
    # Modern health fake news
    ("URGENT: New vaccine contains AI nanobots that can be controlled by 5G signals!", "FAKE"),
    # Legitimate health news
    ("The FDA approved a new diagnostic tool after successful clinical trials showed improved accuracy.", "REAL"),
    # Election misinformation
    ("RIGGED: AI voting machines switched millions of votes! The entire election was manipulated!", "FAKE"),
    # Legitimate political news
    ("Congress held hearings on artificial intelligence regulation with bipartisan support for oversight.", "REAL"),
]

print("\n" + "-" * 70)
print("Modern Fake News Detection Test Results:")
print("-" * 70)

correct = 0
for text, expected in test_samples:
    cleaned = preprocess_text(text)
    vectorized = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    predicted = "FAKE" if prediction == 1 else "REAL"
    confidence = max(probability) * 100
    status = "✅" if predicted == expected else "❌"
    if predicted == expected:
        correct += 1
    print(f"\n{status} Text: {text[:60]}...")
    print(f"   Expected: {expected} | Predicted: {predicted} | Confidence: {confidence:.1f}%")

print("\n" + "-" * 70)
print(f"Test Accuracy: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.0f}%)")
print("-" * 70)

# ============================================
# FINISH
# ============================================
print("\n" + "=" * 70)
print("🎉 COMBINED DATASET TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Dataset: ISOT + Modern 2024-2025 Patterns")
print(f"📊 Total Articles: {len(df_combined):,}")
print(f"📊 TF-IDF Features: {X_train_tfidf.shape[1]:,}")
print(f"📊 Final Accuracy: {accuracy*100:.2f}%")
print("\n🚀 The model can now detect:")
print("   ✓ Traditional fake news patterns")
print("   ✓ AI/Deepfake misinformation")
print("   ✓ Election fraud claims")
print("   ✓ Health scams and conspiracy theories")
print("   ✓ Social media viral misinformation")
print("\n🚀 Run 'python app.py' to start the server!")
print("=" * 70)
