import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os


API_URL = "http://127.0.0.1:8000"
HISTORY_FILE = "data/prediction_history.json"


def load_history() -> list:
    """
    Load prediction history from JSON file.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: list) -> None:
    """
    Save prediction history to JSON file.
    """
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def predict(text: str) -> dict:
    """
    Call the FastAPI predict endpoint.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


st.set_page_config(
    page_title="FinSentinel Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("FinSentinel — French Financial Sentiment Analysis")
st.markdown("*CamemBERT fine-tuned — Accuracy 78.08% — F1 0.7759*")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyse de sentiment")
    text_input = st.text_area(
        "Entrez un texte financier en francais :",
        placeholder="La hausse des taux inquiete les investisseurs...",
        height=120
    )

    if st.button("Analyser", type="primary"):
        if text_input.strip():
            with st.spinner("Analyse en cours..."):
                result = predict(text_input)

            if "error" in result:
                st.error(f"Erreur API : {result['error']}")
            else:
                label = result["label"]
                score = result["score"]
                latency = result["latency_ms"]

                if label == "bullish":
                    st.success(f"BULLISH (optimiste) — Confiance : {score:.1%}")
                elif label == "bearish":
                    st.error(f"BEARISH (pessimiste) — Confiance : {score:.1%}")
                else:
                    st.info(f"NEUTRAL — Confiance : {score:.1%}")

                st.caption(f"Latence : {latency:.0f} ms")

                history = load_history()
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "text": text_input[:100],
                    "label": label,
                    "score": score,
                    "latency_ms": latency
                })
                save_history(history)
        else:
            st.warning("Veuillez entrer un texte.")

with col2:
    st.subheader("Informations modele")
    try:
        info = requests.get(f"{API_URL}/info", timeout=5).json()
        st.metric("Accuracy", f"{info['accuracy']:.1%}")
        st.metric("F1 Score", f"{info['f1_score']:.4f}")
        st.metric("Train samples", f"{info['train_samples']:,}")
    except:
        st.warning("API non disponible")

st.markdown("---")
st.subheader("Historique des predictions")

history = load_history()
if history:
    df = pd.DataFrame(history)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Total predictions", len(df))
    with col4:
        bullish_pct = len(df[df["label"] == "bullish"]) / len(df) * 100
        st.metric("Bullish", f"{bullish_pct:.1f}%")
    with col5:
        bearish_pct = len(df[df["label"] == "bearish"]) / len(df) * 100
        st.metric("Bearish", f"{bearish_pct:.1f}%")

    fig = px.pie(
        df,
        names="label",
        title="Distribution des sentiments",
        color="label",
        color_discrete_map={
            "bullish": "#00cc96",
            "bearish": "#ef553b",
            "neutral": "#636efa"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="timestamp",
        y="latency_ms",
        title="Latence des predictions (ms)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df[["timestamp", "text", "label", "score", "latency_ms"]].tail(10))
else:
    st.info("Aucune prediction encore. Analysez un texte pour commencer.")

st.markdown("---")
st.caption("FinSentinel — Ahmed Walid BOUANZOUL — AI Engineer")