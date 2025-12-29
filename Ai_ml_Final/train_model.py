"""
SENTIMENT ANALYSIS MODEL TRAINING
Script ini akan train model sentiment analysis dari CSV
Author: AI Assistant
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 SENTIMENT ANALYSIS MODEL TRAINING")
print("="*60)

# =========================
# 1. LOAD DATA
# =========================
print("\n[1/8] 📂 Loading data...")

try:
    df = pd.read_csv('sentiment_NetizenIndonesianLangugage.csv')
    print(f"✅ Data loaded: {len(df)} reviews")
except FileNotFoundError:
    print("❌ ERROR: File 'data_training.csv' tidak ditemukan!")
    print("💡 Pastikan file CSV ada di folder yang sama dengan script ini")
    print("💡 Format CSV harus punya kolom: review_text, sentiment")
    exit()

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# kalau sentiment berupa angka
if df['sentiment'].dtype != object:
    df['sentiment'] = df['sentiment'].map(label_map)
    print("✅ Sentiment label dikonversi dari angka ke teks")

# Auto-detect kolom review
review_col = None
for col in ['review_text', 'content', 'review', 'ulasan']:
    if col in df.columns:
        review_col = col
        break

if review_col is None:
    print(f"❌ ERROR: Kolom review tidak ditemukan!")
    print(f"📋 Kolom yang tersedia: {df.columns.tolist()}")
    exit()

if review_col != 'review_text':
    df['review_text'] = df[review_col]
    print(f"✅ Menggunakan kolom '{review_col}' sebagai review text")

# Validasi kolom sentiment
if 'sentiment' not in df.columns:
    print("❌ ERROR: Kolom 'sentiment' tidak ditemukan!")
    print("💡 CSV harus punya kolom 'sentiment' dengan nilai: positive, negative, neutral")
    exit()

# Cek distribusi
print("\n📊 Distribusi Sentiment:")
dist = df['sentiment'].value_counts()
print(dist)

total = len(df)
for sentiment, count in dist.items():
    pct = (count/total)*100
    print(f"   {sentiment}: {count} ({pct:.1f}%)")

if len(df) < 100:
    print("\n⚠️  WARNING: Data terlalu sedikit (<100 reviews)")
    print("💡 Minimal 500 reviews untuk hasil yang baik")
    cont = input("Lanjutkan? (y/n): ")
    if cont.lower() != 'y':
        exit()

# =========================
# 2. PREPROCESSING
# =========================
print("\n[2/8] 🧹 Preprocessing text...")

def clean_text(text):
    """Clean text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s!?.,]', '', text)
    text = ' '.join(text.split())
    return text

def handle_slang(text):
    """Convert slang ke formal"""
    slang_dict = {
        'bgt': 'banget', 'bgus': 'bagus', 'mantap': 'mantap',
        'kecewa': 'kecewa', 'ga': 'tidak', 'gak': 'tidak',
        'nggak': 'tidak', 'makasih': 'terima kasih',
        'thanks': 'terima kasih', 'recomended': 'recommended',
        'keren': 'bagus', 'jelek': 'jelek', 'parah': 'jelek',
        'lambat': 'lambat', 'lama': 'lambat', 'cepat': 'cepat',
        'top': 'bagus', 'mantul': 'mantap', 'zonk': 'jelek',
    }
    
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    return ' '.join(words)

# Apply preprocessing
df['review_clean'] = df['review_text'].apply(clean_text)
df['review_clean'] = df['review_clean'].apply(handle_slang)

# Hapus review terlalu pendek
df = df[df['review_clean'].str.split().str.len() >= 3]
print(f"✅ After cleaning: {len(df)} reviews")

# =========================
# 3. SPLIT DATA
# =========================
print("\n[3/8] ✂️  Splitting data (80% train, 20% test)...")

X = df['review_clean']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# =========================
# 4. FEATURE EXTRACTION
# =========================
print("\n[4/8] 🔢 Extracting features with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
    strip_accents='unicode'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Feature shape: {X_train_vec.shape}")

# =========================
# 5. HANDLE IMBALANCE
# =========================
print("\n[5/8] ⚖️  Handling class imbalance with SMOTE...")

print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))

smote = SMOTE(random_state=42, k_neighbors=min(3, min(pd.Series(y_train).value_counts())-1))
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)

print("After SMOTE:", dict(pd.Series(y_train_balanced).value_counts()))

# =========================
# 6. TRAIN MODELS
# =========================
print("\n[6/8] 🤖 Training models...")

models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(
        max_iter=2000, C=10, solver='lbfgs', 
        class_weight='balanced', random_state=42
    ),
    'Linear SVM': LinearSVC(
        C=1.0, max_iter=2000, 
        class_weight='balanced', random_state=42
    )
}

results = {}
best_accuracy = 0
best_model_name = None
best_model = None

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                  cv=5, scoring='f1_weighted')
    print(f"   ✅ CV F1 Score: {cv_scores.mean():.4f}")
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'y_pred': y_pred
    }
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

# =========================
# 7. BEST MODEL EVALUATION
# =========================
print("\n[7/8] 🏆 Best Model Evaluation")
print("="*60)
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*60)

y_pred_best = results[best_model_name]['y_pred']

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_best))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# =========================
# 8. SAVE MODEL
# =========================
print("\n[8/8] 💾 Saving model...")

# Buat folder model jika belum ada
os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model saved successfully!")
print("   📁 model/sentiment_model.pkl")
print("   📁 model/vectorizer.pkl")

# =========================
# TEST PREDICTIONS
# =========================
print("\n" + "="*60)
print("🧪 TESTING PREDICTIONS")
print("="*60)

test_reviews = [
    "Barang bagus sekali, cepat sampai, recommended!",
    "Mengecewakan, tidak sesuai deskripsi",
    "Biasa saja, tidak istimewa",
    "Kualitas oke, harga terjangkau",
    "Jelek parah, rugi beli ini"
]

for review in test_reviews:
    cleaned = handle_slang(clean_text(review))
    vec = vectorizer.transform([cleaned])
    pred = best_model.predict(vec)[0]
    
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(vec)[0]
        confidence = max(proba) * 100
        print(f"\n📝 {review}")
        print(f"   → {pred.upper()} (Confidence: {confidence:.1f}%)")
    else:
        print(f"\n📝 {review}")
        print(f"   → {pred.upper()}")

# =========================
# SUMMARY
# =========================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"📊 Final Accuracy: {best_accuracy*100:.2f}%")
print(f"🤖 Best Model: {best_model_name}")
print(f"📁 Model saved to: model/sentiment_model.pkl")
print("\n💡 Next step: Gunakan model ini di Streamlit app Anda")
print("="*60)