import joblib
import re

print("="*60)
print("🧪 TESTING SENTIMENT MODEL")
print("="*60)

# Load model
print("\n📂 Loading model...")
try:
    model = joblib.load('model/sentiment_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Model files not found!")
    print("💡 Run 'train_model.py' first to create the model")
    exit()

# Preprocessing functions (sama seperti saat training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s!?.,]', '', text)
    text = ' '.join(text.split())
    return text

def handle_slang(text):
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

# Test reviews
print("\n" + "="*60)
print("📝 TESTING REVIEWS")
print("="*60)

test_reviews = [
    "Barang bagus sekali, puas banget!",
    "Jelek parah, mengecewakan",
    "Biasa saja, tidak istimewa",
    "Kualitas oke, harga terjangkau",
    "Pengiriman cepat, mantap!",
    "Tidak sesuai deskripsi, zonk",
    "Lumayan lah",
    "Recomended banget, top!",
    "Kecewa bgt, rugi beli ini",
    "Bagus, tapi packagingnya jelek"
]

sentiment_emoji = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😡"
}

for i, review in enumerate(test_reviews, 1):
    # Preprocess
    cleaned = handle_slang(clean_text(review))
    
    # Vectorize
    vec = vectorizer.transform([cleaned])
    
    # Predict
    pred = model.predict(vec)[0]  # Ambil elemen pertama
    
    # Get confidence (jika model support predict_proba)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        conf = max(proba) * 100
        emoji = sentiment_emoji.get(pred, "❓")
        print(f"\n{i}. {review}")
        print(f"   → {emoji} {pred.upper()} (Confidence: {conf:.1f}%)")
    else:
        emoji = sentiment_emoji.get(pred, "❓")
        print(f"\n{i}. {review}")
        print(f"   → {emoji} {pred.upper()}")

print("\n" + "="*60)
print("✅ TESTING COMPLETE!")
print("="*60)

# Interactive testing
print("\n💡 Ingin test review sendiri? (ketik 'exit' untuk keluar)")
while True:
    user_input = input("\nMasukkan review: ")
    
    if user_input.lower() in ['exit', 'quit', 'keluar', '']:
        print("👋 Bye!")
        break
    
    # Preprocess
    cleaned = handle_slang(clean_text(user_input))
    
    # Vectorize & Predict
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        conf = max(proba) * 100
        emoji = sentiment_emoji.get(pred, "❓")
        print(f"   → {emoji} {pred.upper()} (Confidence: {conf:.1f}%)")
    else:
        emoji = sentiment_emoji.get(pred, "❓")
        print(f"   → {emoji} {pred.upper()}")