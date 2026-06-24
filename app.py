import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re

# =========================
# LOAD MODELS
# =========================
cnn_model = tf.keras.models.load_model("cnn_anomaly_detector.keras")
lstm_model = tf.keras.models.load_model("lstm_anomaly_detector.keras")
cnn_lstm_model = tf.keras.models.load_model("cnn_lstm_anomaly_detector.keras")

# =========================
# LOAD TOKENIZER
# =========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

MAX_LEN = config["max_sequence_length"]

# =========================
# STREAMLIT UI
# =========================
st.title("🔥 AIOps Log Anomaly Detection System")

model_choice = st.selectbox(
    "Pilih Model",
    ["CNN", "LSTM", "CNN-LSTM"]
)

log_input = st.text_area("Masukkan Log:")

# =========================
# PREPROCESS FUNCTION
# =========================
def clean_log_text(text):
    text = str(text).lower()
    text = re.sub(r'0x[0-9a-f]+', ' HEXADDR ', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' IPADDR ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    cleaned = clean_log_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    return padded

# =========================
# PREDICTION
# =========================
def predict(text, model):
    data = preprocess(text)
    pred = model.predict(data)[0][0]
    return pred

if st.button("Detect Anomaly"):

    if model_choice == "CNN":
        model = cnn_model
    elif model_choice == "LSTM":
        model = lstm_model
    else:
        model = cnn_lstm_model

    score = predict(log_input, model)

    st.subheader("Hasil Prediksi")

    if score > 0.5:
        st.error(f"🚨 ANOMALY (confidence: {score:.3f})")
    else:
        st.success(f"✅ NORMAL (confidence: {1-score:.3f})")

    st.progress(float(score))