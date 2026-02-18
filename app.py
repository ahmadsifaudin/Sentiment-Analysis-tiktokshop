import streamlit as st
import joblib
import numpy as np
import os
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimen Analisis",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #1a0a2e 0%, #0a0a0f 50%, #0d1a0a 100%);
    min-height: 100vh;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.hero-sub {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.05em;
}

/* Card */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    animation: fadeSlide 0.5s ease forwards;
}

.result-card.positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.05));
    border-color: rgba(52,211,153,0.3);
}
.result-card.negative {
    background: linear-gradient(135deg, rgba(248,113,113,0.12), rgba(239,68,68,0.05));
    border-color: rgba(248,113,113,0.3);
}
.result-card.neutral {
    background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(139,92,246,0.05));
    border-color: rgba(167,139,250,0.3);
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.positive .result-label  { color: #34d399; }
.negative .result-label  { color: #f87171; }
.neutral  .result-label  { color: #a78bfa; }

.result-model {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.6rem;
}

.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.conf-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}
.positive .conf-bar { background: linear-gradient(90deg, #34d399, #6ee7b7); }
.negative .conf-bar { background: linear-gradient(90deg, #f87171, #fca5a5); }
.neutral  .conf-bar { background: linear-gradient(90deg, #a78bfa, #c4b5fd); }

.conf-label {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}

/* Divider badge */
.model-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 99px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Textarea */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #000000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    resize: vertical !important;
    padding: 1rem !important;
    transition: border 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.1) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* History */
.history-item {
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #9ca3af;
}
.history-item strong { color: #e8e8f0; }

/* Emoji sentimen */
.sentiment-emoji {
    font-size: 2.5rem;
    line-height: 1;
    margin-bottom: 0.2rem;
}

/* Consensus box */
.consensus-box {
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.5rem;
}
.consensus-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.consensus-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}
.consensus-positive { color: #34d399; }
.consensus-negative { color: #f87171; }
.consensus-neutral  { color: #a78bfa; }

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all model files. Adjust paths if needed."""
    base = os.path.dirname(__file__)

    def load(filename):
        # Try same directory first, then models/ subfolder
        paths = [
            os.path.join(base, filename),
            os.path.join(base, "models", filename),
        ]
        for p in paths:
            if os.path.exists(p):
                return joblib.load(p)
        raise FileNotFoundError(
            f"File '{filename}' tidak ditemukan.\n"
            f"Pastikan file .pkl berada di folder yang sama dengan app.py, "
            f"atau di subfolder 'models/'."
        )

    vectorizer = load("tfidf_vectorizer.pkl")
    nb_model   = load("naive_bayes_model.pkl")
    scaler     = load("scaler.pkl")
    svm_model  = load("svm_model.pkl")
    return vectorizer, nb_model, scaler, svm_model


def predict(text: str, vectorizer, nb_model, scaler, svm_model):
    """Return predictions and probabilities from both models."""
    X_tfidf = vectorizer.transform([text])

    # Naive Bayes
    nb_pred  = nb_model.predict(X_tfidf)[0]
    nb_proba = nb_model.predict_proba(X_tfidf)[0]
    nb_conf  = float(np.max(nb_proba))

    # SVM (needs scaled input)
    X_scaled = scaler.transform(X_tfidf)
    svm_pred = svm_model.predict(X_scaled)[0]

    # SVM confidence: decision_function if no proba
    try:
        svm_proba = svm_model.predict_proba(X_scaled)[0]
        svm_conf  = float(np.max(svm_proba))
    except AttributeError:
        df = svm_model.decision_function(X_scaled)[0]
        # Softmax approximation
        exp_df    = np.exp(df - np.max(df))
        svm_proba = exp_df / exp_df.sum()
        svm_conf  = float(np.max(svm_proba))

    classes = nb_model.classes_.tolist()

    return {
        "nb":  {"label": nb_pred,  "conf": nb_conf,  "proba": dict(zip(classes, nb_proba.tolist()))},
        "svm": {"label": svm_pred, "conf": svm_conf, "proba": dict(zip(classes, svm_proba.tolist()))},
    }


def sentiment_meta(label: str):
    mapping = {
        "Positive": ("positive", "😊", "Positif"),
        "Negative": ("negative", "😞", "Negatif"),
        "Neutral":  ("neutral",  "😐", "Netral"),
    }
    return mapping.get(label, ("neutral", "❓", label))


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
# LOAD MODELS (with user-friendly error)
# ─────────────────────────────────────────────
try:
    vectorizer, nb_model, scaler, svm_model = load_models()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    load_error = str(e)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">Sentimen Analisis</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Naive Bayes &nbsp;·&nbsp; Support Vector Machine &nbsp;·&nbsp; TF-IDF</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Gagal memuat model:\n\n{load_error}")
    st.stop()

# ── Input ─────────────────────────────────────
text_input = st.text_area(
    label="Masukkan teks",
    placeholder="Tulis ulasan, komentar, atau teks lainnya di sini…",
    height=140,
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Bersihkan", use_container_width=True)

if clear_btn:
    st.session_state.history = []
    st.rerun()

# ── Prediction ────────────────────────────────
if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis…"):
            time.sleep(0.3)   # UX pause
            results = predict(text_input.strip(), vectorizer, nb_model, scaler, svm_model)

        # Determine consensus
        nb_lbl  = results["nb"]["label"]
        svm_lbl = results["svm"]["label"]
        if nb_lbl == svm_lbl:
            consensus = nb_lbl
        else:
            # Prefer higher confidence
            consensus = nb_lbl if results["nb"]["conf"] >= results["svm"]["conf"] else svm_lbl

        c_css, c_emoji, c_text = sentiment_meta(consensus)

        # Consensus
        st.markdown(f"""
        <div class="consensus-box">
            <div class="consensus-title">Hasil Keseluruhan</div>
            <div style="font-size:2.8rem; margin:0.2rem 0;">{c_emoji}</div>
            <div class="consensus-value consensus-{c_css}">{c_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Two-column model results
        col_nb, col_svm = st.columns(2)

        for col, model_key, model_name in [
            (col_nb,  "nb",  "Naive Bayes"),
            (col_svm, "svm", "SVM"),
        ]:
            res = results[model_key]
            css, emoji, text_id = sentiment_meta(res["label"])
            conf_pct = int(res["conf"] * 100)

            with col:
                st.markdown(f"""
                <div class="result-card {css}">
                    <div class="result-model">{model_name}</div>
                    <div class="sentiment-emoji">{emoji}</div>
                    <div class="result-label">{text_id}</div>
                    <div class="conf-label">Keyakinan &nbsp;{conf_pct}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{conf_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Probability breakdown (expander)
        with st.expander("📊 Lihat distribusi probabilitas"):
            import pandas as pd
            classes = list(results["nb"]["proba"].keys())
            df_proba = pd.DataFrame({
                "Kelas": classes,
                "Naive Bayes (%)": [round(results["nb"]["proba"][c]*100, 2) for c in classes],
                "SVM (%)":         [round(results["svm"]["proba"][c]*100, 2) for c in classes],
            })
            st.dataframe(df_proba, hide_index=True, use_container_width=True)

        # Save to history
        st.session_state.history.insert(0, {
            "text":      text_input.strip()[:80] + ("…" if len(text_input.strip()) > 80 else ""),
            "nb":        nb_lbl,
            "svm":       svm_lbl,
            "consensus": consensus,
        })
        st.session_state.history = st.session_state.history[:10]  # keep last 10


# ── History ───────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<span class="model-badge">Riwayat Analisis</span>', unsafe_allow_html=True)

    for item in st.session_state.history:
        _, emoji, text_id = sentiment_meta(item["consensus"])
        st.markdown(f"""
        <div class="history-item">
            {emoji} <strong>{text_id}</strong> &nbsp;·&nbsp;
            NB: {item['nb']} &nbsp;|&nbsp; SVM: {item['svm']}<br>
            <span style="color:#6b7280; font-size:0.8rem;">"{item['text']}"</span>
        </div>
        """, unsafe_allow_html=True)
