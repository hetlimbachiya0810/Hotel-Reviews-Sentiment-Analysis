import streamlit as st
import numpy as np
import pickle
import os
from keras.preprocessing import sequence

# --------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Hotel Review Sentiment Analysis",
    page_icon="🏨",
    layout="centered"
)

# --------------------------------------------
# 2. LOAD MODEL & VOCAB
# --------------------------------------------

@st.cache_resource
def load_model():
    model_path = "hotel-sentiments-model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_word_index():
    vocab_path = "word_to_int.pkl"
    if not os.path.exists(vocab_path):
        st.error(f"❌ Vocabulary file not found: {vocab_path}")
        st.stop()

    with open(vocab_path, "rb") as f:
        word_to_int = pickle.load(f)

    return word_to_int


model = load_model()
word_to_int = load_word_index()

MAX_LEN = 120  # from your notebook


# --------------------------------------------
# 3. TEXT → SEQUENCE PREPROCESSING FUNCTIONS
# --------------------------------------------

def convert_review_to_word_id(raw_review):
    ids = []
    raw_review = raw_review.lower()

    for word in raw_review.split():
        if word in word_to_int:
            ids.append(word_to_int[word])
        else:
            ids.append(0)
    return ids


def preprocess_review(text):
    word_ids = convert_review_to_word_id(text)
    padded = sequence.pad_sequences([word_ids], maxlen=MAX_LEN, padding="post")
    return padded


# --------------------------------------------
# 4. PREDICTION FUNCTION
# --------------------------------------------

def predict_sentiment(text):
    processed = preprocess_review(text)
    pred = model.predict(processed)[0]   # model outputs [positive_prob, negative_prob]

    positive_prob = float(pred[1])
    negative_prob = float(pred[0])

    sentiment = "Positive" if positive_prob >= negative_prob else "Negative"
    confidence = max(positive_prob, negative_prob)

    return sentiment, confidence


# --------------------------------------------
# 5. CUSTOM CSS (Clean UI)
# --------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.result-card {
    padding: 1rem 1.2rem;
    border-radius: 10px;
    background: #fafafa;
    border: 1px solid #eee;
    margin-top: 1rem;
}
.badge-positive {
    padding: 0.4rem 1rem;
    background: #d1fae5;
    color: #065f46;
    border-radius: 50px;
    font-weight: bold;
}
.badge-negative {
    padding: 0.4rem 1rem;
    background: #fee2e2;
    color: #991b1b;
    border-radius: 50px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
# 6. UI LAYOUT
# --------------------------------------------

st.markdown("<div class='main-title'>🏨 Hotel Review Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning Model • Positive / Negative Classification</div>", unsafe_allow_html=True)

review_text = st.text_area(
    "✍️ Enter a hotel review",
    height=150,
    placeholder="Type your hotel review here…"
)

if st.button("🔍 Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing…"):
            label, confidence = predict_sentiment(review_text)

        st.markdown("### 📊 Prediction Result")
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        if label == "Positive":
            st.markdown("<span class='badge-positive'>😊 Positive</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-negative'>☹️ Negative</span>", unsafe_allow_html=True)

        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("<br><center><sub>Created using Streamlit • Deep Learning Model</sub></center>", unsafe_allow_html=True)
