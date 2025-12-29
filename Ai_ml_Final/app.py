import streamlit as st
import pandas as pd
import joblib
import os
import re
import numpy as np
import plotly.graph_objects as go
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ReviewSense",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL STYLE (COLORFUL)
# =========================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background: linear-gradient(135deg, #fff0f6, #f3e5ff);
}
.card {
    background: linear-gradient(135deg, #ffffff, #fff7fb);
    padding: 22px;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(255,77,141,0.25);
    margin-bottom: 25px;
}
.title {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4d8d, #8e44ad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #666;
    font-size: 15px;
}
.metric-title {
    font-size: 14px;
    color: #777;
}
.metric-value {
    font-size: 34px;
    font-weight: 800;
    color: #ff4d8d;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    background: #ffe3ec;
    color: #ff4d8d;
}
.notice {
    background: linear-gradient(135deg, #fff3cd, #ffe8a1);
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("""
<h1 style='color:#ff4d8d'>💗 ReviewSense</h1>
<p style='color:#555'>AI Review & Reputation Analyzer</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))
    st.sidebar.success("✅ Model AI siap digunakan")
except:
    st.sidebar.error("❌ Model tidak ditemukan")
    st.stop()

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s!?.,]', '', text)
    return ' '.join(text.split())

def handle_slang(text):
    slang = {
        'bgt':'banget','ga':'tidak','gak':'tidak','nggak':'tidak',
        'makasih':'terima kasih','thanks':'terima kasih',
        'keren':'bagus','mantul':'mantap','zonk':'jelek','parah':'jelek'
    }
    return ' '.join([slang.get(w, w) for w in text.split()])

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card">
    <div class="title">💬 ReviewSense Dashboard</div>
    <p class="subtitle">
        Analisis sentimen otomatis dari ribuan review hanya dengan upload CSV
    </p>
    <span class="badge">Machine Learning</span>
    <span class="badge">Sentiment Analysis</span>
    <span class="badge">CSV Upload</span>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD NOTICE
# =========================
st.markdown("""
<div class="notice">
<h4>⚠️ Ketentuan Upload File CSV (WAJIB DIBACA)</h4>
<ul>
<li>📄 Format file: <b>.CSV</b></li>
<li>📝 Wajib memiliki <b>1 kolom review</b></li>
<li>✅ Nama kolom yang didukung:
    <code>review_text</code>,
    <code>content</code>,
    <code>review</code>,
    <code>ulasan</code>
</li>
<li>❌ File <b>Excel (.xlsx)</b> tidak didukung</li>
<li>💡 Disarankan minimal <b>10 review</b></li>
</ul>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD CSV
# =========================
uploaded_file = st.file_uploader("📂 Upload File CSV Review", type=["csv"])

if not uploaded_file:
    st.info("⬆️ Silakan upload file CSV untuk memulai analisis")
    st.stop()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(uploaded_file, engine="python")

possible_cols = ['review_text','content','review','ulasan','text','comment']
review_col = next((c for c in possible_cols if c in df.columns), None)

if review_col is None:
    st.error("❌ Kolom review tidak ditemukan di CSV Anda")
    st.write("Kolom tersedia:", list(df.columns))
    st.stop()

df["review_text"] = df[review_col].astype(str)

# =========================
# ANALYSIS
# =========================
with st.spinner("🤖 AI sedang membaca emosi netizen..."):
    df["clean"] = df["review_text"].apply(clean_text).apply(handle_slang)
    X = vectorizer.transform(df["clean"])
    df["sentiment"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        df["confidence"] = np.max(model.predict_proba(X), axis=1) * 100
    else:
        df["confidence"] = np.nan

# =========================
# METRICS
# =========================
count = df["sentiment"].value_counts()
pos, neu, neg = count.get("positive",0), count.get("neutral",0), count.get("negative",0)
total = pos + neu + neg
health = round((pos/total)*100,1) if total else 0

c1,c2,c3,c4 = st.columns(4)

def metric(col, title, value, emoji):
    with col:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">{emoji} {title}</div>
        <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric(c1,"Positive",pos,"😊")
metric(c2,"Neutral",neu,"😐")
metric(c3,"Negative",neg,"😡")
metric(c4,"Health Score",f"{health}%","❤️")

# =========================
# PIE CHART
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
fig = go.Figure(go.Pie(
    labels=["Positive","Neutral","Negative"],
    values=[pos,neu,neg],
    hole=0.6,
    marker=dict(colors=["#00c853","#ffca28","#f44336"])
))
fig.update_layout(title="Distribusi Sentimen AI", height=420)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DETAIL TABLE
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Detail Review & Prediksi")

show_df = df[["review_text","sentiment","confidence"]]
show_df.columns = ["Review","Sentiment","Confidence (%)"]

st.dataframe(show_df, use_container_width=True, height=420)

st.download_button(
    "⬇️ Download Hasil Analisis",
    df.to_csv(index=False),
    "review_sense_result.csv",
    "text/csv"
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 ReviewSense • Sentiment Analysis Dashboard • Powered by Machine Learning")
