import joblib
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="KPP App",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/xgb_model_kw_filtered_df.pkl"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
TARGET_COLS = ["click_count", "keyword_rps"]

EMBEDDING_DIM = 384


@st.cache_resource
def load_models_and_device():
    """Loads the SentenceTransformer and XGBoost models, and determines the device."""
    embedder_model = None
    xgboost_pipeline_model = None
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    st.sidebar.info(f"Using device: {compute_device.upper()}")

    try:
        embedder_model = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL, device=compute_device
        )
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading Sentence Transformer: {str(e)[:100]}...")
        st.error(
            f"Critical Error: Could not load Sentence Transformer model. Details: {e}"
        )
        return None, None, None

    try:
        xgboost_pipeline_model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.sidebar.error(f"⚠️ XGBoost model file '{MODEL_PATH}' not found.")
        st.error(
            f"Critical Error: XGBoost model file '{MODEL_PATH}' not found. Please ensure it's in the correct location."
        )
        return embedder_model, None, compute_device
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading XGBoost model: {str(e)[:100]}...")
        st.error(f"Critical Error: Could not load XGBoost model. Details: {e}")
        return embedder_model, None, compute_device

    return embedder_model, xgboost_pipeline_model, compute_device


embedder, xgb_model, device = load_models_and_device()


def predict_metrics_from_keyword(keyword_text: str):
    """
    Generates embeddings for the keyword and predicts performance metrics.
    """
    if not xgb_model or not embedder:
        st.error("Models are not fully loaded. Cannot perform prediction.")
        return None

    if not keyword_text.strip():
        return None

    embedding = embedder.encode([keyword_text], device=device, show_progress_bar=False)
    embedding_df = pd.DataFrame(
        embedding, columns=[f"embedding_{i}" for i in range(EMBEDDING_DIM)]
    )

    try:
        predictions_array = xgb_model.predict(embedding_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    results = dict(zip(TARGET_COLS, predictions_array[0]))
    return results


_, center_col, _ = st.columns([0.1, 0.5, 0.1])
with center_col:
    st.image("assets/logo.png")

st.html(
    """
    <div style="text-align: center;">
        Welcome! This app uses advanced AI models to predict potential performance metrics for a given keyword.
        <br>
        Simply enter your keyword and see the estimated results.
    </div>
    """
)

st.sidebar.header("About the Metrics")
st.sidebar.markdown(
    """
- **Search Count:** An estimate of how many times the keyword might be searched.
- **Click Count:** An estimate of how many clicks ads related to this keyword might receive.
- **Unique Clicks:** An estimate of the number of distinct users clicking on the ads.
- **Avg RPC (Revenue Per Click):** An estimate of the average revenue generated per click.
"""
)
st.sidebar.markdown("---")
if embedder and xgb_model:
    st.sidebar.success("All models loaded successfully.")
elif embedder:
    st.sidebar.warning(
        "XGBoost model failed to load. Prediction functionality is disabled."
    )
elif xgb_model:
    st.sidebar.warning(
        "Sentence Transformer model failed to load. Prediction functionality is disabled."
    )
else:
    st.sidebar.error("Critical model loading failure. App may not function.")

if xgb_model and embedder:
    keyword_input = st.text_input(
        "🔑 Enter your keyword below:",
        placeholder="e.g., sustainable energy solutions, best travel cameras 2024",
        help="Type the keyword you want to analyze.",
    )

    if st.button("🚀 Predict Performance", type="primary", use_container_width=True):
        if keyword_input.strip():
            with st.spinner("🧠 Analyzing keyword and crunching numbers..."):
                predicted_values = predict_metrics_from_keyword(keyword_input)

            if predicted_values:
                st.subheader("📈 Predicted Metrics")

                cols = st.columns(len(TARGET_COLS))
                for i, (metric_name, value) in enumerate(predicted_values.items()):
                    display_name = metric_name.replace("_", " ").title()

                    if "count" in metric_name:
                        formatted_val = f"{float(value):.2f}"
                    elif "rpc" in metric_name:
                        formatted_val = f"${float(value):.4f}"
                    else:
                        formatted_val = f"{float(value):.4f}"

                    cols[i].metric(label=display_name, value=formatted_val)

                st.markdown("---")

                with st.expander("📝 View Raw Prediction Output", expanded=False):

                    raw_output_str = (
                        "{\n"
                        + "\n".join(
                            [f"    '{k}': {v}," for k, v in predicted_values.items()]
                        )
                        + "\n}"
                    )

                    st.code(raw_output_str, language="text")

        else:
            st.warning("🔔 Please enter a keyword to get predictions.")
else:
    st.error(
        "🚨 Application critical error: One or more models could not be loaded. Please check the sidebar and console/logs for details. Ensure 'xgb_multioutput_model.pkl' is present and valid."
    )

st.markdown("---")
