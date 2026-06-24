import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re

# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_models():
    bgl = {
        "CNN":      tf.keras.models.load_model("cnn_bgl.keras"),
        "LSTM":     tf.keras.models.load_model("lstm_bgl.keras"),
        "CNN-LSTM": tf.keras.models.load_model("cnn_lstm_bgl.keras"),
    }
    hdfs = {
        "CNN":      tf.keras.models.load_model("cnn_hdfs.keras"),
        "LSTM":     tf.keras.models.load_model("lstm_hdfs.keras"),
        "CNN-LSTM": tf.keras.models.load_model("cnn_lstm_hdfs.keras"),
    }
    return bgl, hdfs

# =============================================================================
# LOAD TOKENIZERS & CONFIGS
# =============================================================================
@st.cache_resource
def load_artifacts():
    with open("tokenizer_bgl.pkl", "rb") as f:
        tok_bgl = pickle.load(f)
    with open("tokenizer_hdfs.pkl", "rb") as f:
        tok_hdfs = pickle.load(f)
    with open("model_config_bgl.pkl", "rb") as f:
        cfg_bgl = pickle.load(f)
    with open("model_config_hdfs.pkl", "rb") as f:
        cfg_hdfs = pickle.load(f)
    return tok_bgl, tok_hdfs, cfg_bgl, cfg_hdfs

bgl_models, hdfs_models        = load_models()
tok_bgl, tok_hdfs, cfg_bgl, cfg_hdfs = load_artifacts()

MAX_LEN_BGL  = cfg_bgl["max_sequence_length"]
MAX_LEN_HDFS = cfg_hdfs["max_sequence_length"]

# =============================================================================
# PREPROCESSING
# =============================================================================
def clean_log_text(text):
    text = str(text).lower()
    text = re.sub(r'0x[0-9a-f]+', ' HEXADDR ', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' IPADDR ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text, tokenizer, max_len):
    cleaned = clean_log_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = tf.keras.utils.pad_sequences(seq, maxlen=max_len,
                                           padding='post', truncating='post')
    return padded

# =============================================================================
# ENSEMBLE PREDICTION
# =============================================================================
def ensemble_predict(log_text):
    detail = []

    for model_name, model in bgl_models.items():
        data  = preprocess(log_text, tok_bgl, MAX_LEN_BGL)
        score = float(model.predict(data, verbose=0)[0][0])
        detail.append({
            "Dataset": "BGL",
            "Model":   model_name,
            "Score":   score,
            "Label":   "ANOMALY" if score > 0.5 else "NORMAL",
        })

    for model_name, model in hdfs_models.items():
        data  = preprocess(log_text, tok_hdfs, MAX_LEN_HDFS)
        score = float(model.predict(data, verbose=0)[0][0])
        detail.append({
            "Dataset": "HDFS",
            "Model":   model_name,
            "Score":   score,
            "Label":   "ANOMALY" if score > 0.5 else "NORMAL",
        })

    # Per-dataset majority vote (2/3 model setuju → dataset flagged)
    bgl_votes  = sum(1 for d in detail if d["Dataset"] == "BGL"  and d["Label"] == "ANOMALY")
    hdfs_votes = sum(1 for d in detail if d["Dataset"] == "HDFS" and d["Label"] == "ANOMALY")
    bgl_flagged  = bgl_votes  >= 2
    hdfs_flagged = hdfs_votes >= 2

    # Final: ANOMALY jika salah satu dataset majority flagged (OR logic)
    final_label = "ANOMALY" if (bgl_flagged or hdfs_flagged) else "NORMAL"
    avg_score   = float(np.mean([d["Score"] for d in detail]))
    n_anomaly   = sum(1 for d in detail if d["Label"] == "ANOMALY")

    return detail, avg_score, n_anomaly, final_label, bgl_votes, hdfs_votes

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="AIOps Log Anomaly Detection", page_icon="🔥", layout="wide")

st.title("🔥 AIOps Log Anomaly Detection System")
st.caption("Ensemble: BGL + HDFS | Models: CNN · LSTM · CNN-LSTM")

with st.expander("ℹ️ Informasi Model & Artifacts", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**BGL (BlueGene/L Supercomputer)**")
        st.code(
            "cnn_bgl.keras\nlstm_bgl.keras\ncnn_lstm_bgl.keras\n"
            "tokenizer_bgl.pkl\nmodel_config_bgl.pkl"
        )
    with col_b:
        st.markdown("**HDFS (Hadoop Distributed File System)**")
        st.code(
            "cnn_hdfs.keras\nlstm_hdfs.keras\ncnn_lstm_hdfs.keras\n"
            "tokenizer_hdfs.pkl\nmodel_config_hdfs.pkl"
        )

st.divider()

log_input = st.text_area(
    "Masukkan Log:",
    placeholder="Contoh: kernel: BUG: unable to handle kernel NULL pointer dereference",
    height=120,
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    detect = st.button("🔍 Detect Anomaly", use_container_width=True)
with col_info:
    st.info(
        "**Cara kerja ensemble:** semua 6 model (3 BGL + 3 HDFS) diprediksi, "
        "keputusan final berdasarkan rata-rata probabilitas.",
        icon="ℹ️",
    )

if detect:
    if not log_input.strip():
        st.warning("Masukkan teks log terlebih dahulu.")
    else:
        with st.spinner("Menjalankan 6 model..."):
            detail, avg_score, n_anomaly, final_label, bgl_votes, hdfs_votes = ensemble_predict(log_input)

        st.divider()

        # --- Hasil Final ---
        st.subheader("Hasil Ensemble")
        col1, col2, col3 = st.columns(3)

        with col1:
            if final_label == "ANOMALY":
                st.error("🚨 **ANOMALY**")
            else:
                st.success("✅ **NORMAL**")

        with col2:
            st.metric("BGL Votes", f"{bgl_votes} / 3", help="≥2 → flagged")
            st.metric("HDFS Votes", f"{hdfs_votes} / 3", help="≥2 → flagged")

        with col3:
            st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
            st.metric("Total Votes", f"{n_anomaly} / 6 model")

        st.progress(avg_score, text=f"Anomaly score: {avg_score:.4f}")

        # --- Detail per Model ---
        st.divider()
        st.subheader("Detail per Model")

        col_bgl, col_hdfs = st.columns(2)

        with col_bgl:
            st.markdown("**BGL (BlueGene/L)**")
            for d in [x for x in detail if x["Dataset"] == "BGL"]:
                icon = "🚨" if d["Label"] == "ANOMALY" else "✅"
                st.write(f"{icon} **{d['Model']}** — {d['Label']} `{d['Score']:.4f}`")
                st.progress(d["Score"])

        with col_hdfs:
            st.markdown("**HDFS (Hadoop)**")
            for d in [x for x in detail if x["Dataset"] == "HDFS"]:
                icon = "🚨" if d["Label"] == "ANOMALY" else "✅"
                st.write(f"{icon} **{d['Model']}** — {d['Label']} `{d['Score']:.4f}`")
                st.progress(d["Score"])
