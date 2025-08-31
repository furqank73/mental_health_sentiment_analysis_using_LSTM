import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# Load Model, Tokenizer, and Label Encoder
# ===============================
model = load_model("sentiment_model.keras")
tokenizer = joblib.load("tokenizer.pickle")
le = joblib.load("labelencoder.pickle")   # make sure you saved it during training

MAX_LEN = 150  # same maxlen as training

# ===============================
# Prediction Function
# ===============================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(pad, verbose=0)
    label = le.inverse_transform([np.argmax(pred)])
    return label[0], float(np.max(pred))  # return label + confidence


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="🧠 Sentiment Analysis", layout="centered")

st.title("🧠 Sentiment Analysis App")
st.write("Enter a sentence and the model will predict its sentiment.")

# Input box
user_input = st.text_area("✍️ Enter your text:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        label, confidence = predict_sentiment(user_input)
        
        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("⚠️ Please enter some text before predicting.")
