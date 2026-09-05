#!/usr/bin/env python3
"""Streamlit interface for an educational news-text classification system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

from detect_fake_news import classify_probability
from model_compat import load_pipeline as load_model_pipeline


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="NewsCheck",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pipeline_path() -> Path:
    return project_root() / "outputs" / "pipeline.joblib"


def default_metrics_path() -> Path:
    return project_root() / "outputs" / "metrics.json"


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_pipeline(path: str):
    return load_model_pipeline(path)


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


# ============================================================
# HELPERS
# ============================================================

def is_short_input(text: str) -> bool:
    words = text.strip().split()
    return len(words) < 8 or len(text.strip()) < 50


# ============================================================
# MODEL PATH
# ============================================================

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "--pipeline",
    default=str(default_pipeline_path()),
)

args, _ = parser.parse_known_args()

pipeline_path = Path(args.pipeline).resolve()
metrics_path = default_metrics_path()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ NewsCheck")

    st.caption("Intelligent News Text Screening")

    st.divider()

    st.subheader("Prediction Controls")

    threshold = st.slider(
        "Classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Controls the probability level used by the classifier.",
    )

    uncertainty_margin = st.slider(
        "Uncertainty margin",
        min_value=0.00,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Creates an uncertainty zone around the classification threshold.",
    )

    st.divider()

    st.subheader("System Details")

    st.write("**Feature Extraction**")
    st.write("TF-IDF")

    st.write("**Classifier**")
    st.write("Logistic Regression")

    st.write("**Classification**")
    st.write("REAL / FAKE / UNCERTAIN")

    st.divider()

    metrics = load_metrics(metrics_path)

    if metrics:

        test = metrics.get("holdout_test", {})

        st.subheader("Evaluation")

        st.metric(
            "Accuracy",
            f"{test.get('accuracy', 0):.3f}",
        )

        st.metric(
            "Macro F1",
            f"{test.get('macro_f1', 0):.3f}",
        )

        st.metric(
            "ROC-AUC",
            f"{test.get('roc_auc', 0):.3f}",
        )

    st.divider()

    st.caption(
        "For educational use. The prediction indicates "
        "textual similarity to the model's training patterns "
        "and is not a factual verification."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🛡️ NewsCheck")

st.subheader("News Credibility Pattern Analyzer")

st.write(
    "Enter a news headline or article excerpt. "
    "The machine-learning system examines the language "
    "and estimates which category it most closely resembles."
)

st.divider()


# ============================================================
# MODEL CHECK
# ============================================================

if not pipeline_path.exists():

    st.error(
        "The trained classification model is unavailable."
    )

    st.info(
        "Train the model first with "
        "`python src/train_model.py`."
    )

    st.stop()


pipeline = load_pipeline(
    str(pipeline_path)
)


# ============================================================
# NEWS INPUT
# ============================================================

st.subheader("📰 News Text")

st.caption(
    "For better results, provide a complete headline "
    "or several sentences from the article."
)

text = st.text_area(
    "Input",
    height=250,
    placeholder=(
        "Paste the news headline or article excerpt here..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# QUICK INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Threshold",
        f"{threshold:.0%}",
    )

with col2:

    st.metric(
        "Uncertainty",
        f"±{uncertainty_margin / 2:.0%}",
    )

with col3:

    words = (
        len(text.strip().split())
        if text.strip()
        else 0
    )

    st.metric(
        "Word Count",
        words,
    )


st.write("")


# ============================================================
# ANALYZE
# ============================================================

analyze = st.button(
    "🔎 Run Analysis",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not text.strip():

        st.warning(
            "Please enter some news text first."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        fake_probability = float(
            pipeline.predict_proba([text])[0, 1]
        )

    except Exception as error:

        st.error(
            "The system could not process this text."
        )

        st.exception(error)

        st.stop()


    real_probability = 1 - fake_probability


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    prediction = classify_probability(
        fake_probability,
        threshold,
        uncertainty_margin,
    )


    # --------------------------------------------------------
    # DECISION RANGE
    # --------------------------------------------------------

    half_margin = uncertainty_margin / 2

    lower_limit = max(
        0.0,
        threshold - half_margin,
    )

    upper_limit = min(
        1.0,
        threshold + half_margin,
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("📋 Analysis Result")

    if prediction == "FAKE":

        st.error(
            "⚠️ FAKE PATTERN DETECTED"
        )

        st.write(
            "The language of this text is more closely "
            "associated with examples from the FAKE category "
            "in the model's training data."
        )

    elif prediction == "REAL":

        st.success(
            "✅ REAL PATTERN DETECTED"
        )

        st.write(
            "The language of this text is more closely "
            "associated with examples from the REAL category "
            "in the model's training data."
        )

    else:

        st.warning(
            "❓ INCONCLUSIVE RESULT"
        )

        st.write(
            "The prediction is close to the classification "
            "boundary, so the system cannot make a clear "
            "classification."
        )


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Classification",
            prediction,
        )

    with result2:

        st.metric(
            "Fake Score",
            f"{fake_probability:.1%}",
        )

    with result3:

        st.metric(
            "Real Score",
            f"{real_probability:.1%}",
        )


    st.write("")


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader("📊 Probability Distribution")

    st.progress(
        fake_probability,
        text=f"Fake category: {fake_probability:.1%}",
    )

    st.caption(
        f"Classification zones: "
        f"REAL < {lower_limit:.0%} | "
        f"UNCERTAIN {lower_limit:.0%}–{upper_limit:.0%} | "
        f"FAKE > {upper_limit:.0%}"
    )


    # ========================================================
    # TEXT STATISTICS
    # ========================================================

    st.subheader("📌 Text Statistics")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:

        st.metric(
            "Words",
            len(text.strip().split()),
        )

    with stat2:

        st.metric(
            "Characters",
            len(text),
        )

    with stat3:

        st.metric(
            "Model Confidence",
            f"{max(fake_probability, real_probability):.1%}",
        )


    # ========================================================
    # SHORT TEXT
    # ========================================================

    if is_short_input(text):

        st.warning(
            "This input is quite short. "
            "A longer article excerpt may provide the model "
            "with more useful language patterns."
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("💡 Interpretation")

    if prediction == "FAKE":

        st.write(
            "The classifier has identified linguistic patterns "
            "that are statistically closer to the FAKE examples "
            "used during training."
        )

    elif prediction == "REAL":

        st.write(
            "The classifier has identified linguistic patterns "
            "that are statistically closer to the REAL examples "
            "used during training."
        )

    else:

        st.write(
            "The two categories are too close for the system "
            "to confidently select one."
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.info(
        "⚠️ Important: This application is a machine-learning "
        "classification system. It does not independently "
        "verify facts, sources, dates, people, or events. "
        "Always verify important claims using reliable sources."
    )


# ============================================================
# TECHNOLOGY SECTION
# ============================================================

st.divider()

st.subheader("⚙️ Technology Behind the System")

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.write("### 1. Text Representation")

    st.write(
        "The input news text is converted into numerical "
        "features so that the machine-learning algorithm "
        "can process it."
    )


with tech2:

    st.write("### 2. TF-IDF")

    st.write(
        "TF-IDF measures the importance of words and phrases "
        "within the training collection."
    )


with tech3:

    st.write("### 3. Classification")

    st.write(
        "Logistic Regression uses the extracted features "
        "to estimate the probability of each category."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NewsCheck • Educational Machine Learning Application"
)
