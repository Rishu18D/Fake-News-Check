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

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --primary-color: #54f2e3;
        --ink: #edf7ff;
        --muted: #91a5bb;
        --cyan: #54f2e3;
        --blue: #62a8ff;
        --pink: #ff5eae;
        --panel: rgba(12, 23, 40, 0.82);
        --line: rgba(117, 169, 207, 0.17);
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", "Arial", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 78% 0%, rgba(27, 110, 157, 0.17), transparent 33rem),
            radial-gradient(ellipse at 0% 22%, rgba(113, 42, 122, 0.11), transparent 28rem),
            #070b14;
    }

    [data-testid="stAppViewContainer"]::before {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        inset: 0;
        content: "";
        opacity: 0.13;
        background-image:
            linear-gradient(rgba(120, 177, 213, 0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(120, 177, 213, 0.12) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMain"] {
        position: relative;
        z-index: 1;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #080d17 100%);
    }

    [data-testid="stSidebar"] > div:first-child {
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: #adbed0;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ink);
        letter-spacing: 0.02em;
    }

    .sidebar-brand {
        padding: 0.4rem 0 1rem;
    }

    .sidebar-brand-name {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -0.04em;
    }

    .sidebar-brand-name span {
        color: var(--cyan);
    }

    .sidebar-tagline {
        margin-top: 0.25rem;
        color: var(--muted);
        font-size: 0.77rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        margin: 0.3rem 0 1.6rem;
        padding: clamp(1.7rem, 4vw, 3.2rem);
        border: 1px solid rgba(84, 242, 227, 0.22);
        border-radius: 22px;
        background:
            linear-gradient(112deg, rgba(15, 32, 49, 0.96), rgba(12, 17, 33, 0.93) 60%, rgba(42, 19, 48, 0.76));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25), inset 0 1px rgba(255, 255, 255, 0.04);
    }

    .hero-shell::after {
        position: absolute;
        top: -8rem;
        right: -4rem;
        width: 23rem;
        height: 23rem;
        border: 1px solid rgba(84, 242, 227, 0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 2.5rem rgba(84, 242, 227, 0.025), 0 0 0 5rem rgba(98, 168, 255, 0.025);
        content: "";
        pointer-events: none;
    }

    .hero-kicker, .section-kicker {
        color: var(--cyan);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        margin: 0.8rem 0 0.65rem;
        color: var(--ink);
        font-size: clamp(2.15rem, 5vw, 4rem);
        font-weight: 760;
        letter-spacing: -0.065em;
        line-height: 1.04;
    }

    .hero-title span {
        color: var(--cyan);
        text-shadow: 0 0 28px rgba(84, 242, 227, 0.28);
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 670px;
        margin: 0;
        color: #a9bbcf;
        font-size: 1rem;
        line-height: 1.7;
    }

    .hero-status {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin-top: 1.35rem;
        padding: 0.45rem 0.72rem;
        border: 1px solid rgba(84, 242, 227, 0.2);
        border-radius: 999px;
        background: rgba(84, 242, 227, 0.06);
        color: #c4fff7;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 12px var(--cyan);
    }

    .section-kicker {
        margin: 0.5rem 0 0.35rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #a9bbcf;
    }

    [data-testid="stTextArea"] textarea {
        min-height: 240px;
        border: 1px solid rgba(98, 168, 255, 0.26);
        border-radius: 14px;
        background: rgba(7, 13, 24, 0.9);
        color: var(--ink);
        line-height: 1.7;
        box-shadow: inset 0 0 24px rgba(45, 91, 131, 0.08);
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--cyan);
        box-shadow: 0 0 0 1px var(--cyan), 0 0 22px rgba(84, 242, 227, 0.09);
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #657b92;
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(18, 33, 52, 0.88), rgba(11, 19, 33, 0.9));
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #91a5bb;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 700;
    }

    [data-testid="stButton"] > button {
        min-height: 3.15rem;
        border: 1px solid rgba(84, 242, 227, 0.55);
        border-radius: 11px;
        background: linear-gradient(100deg, #1ec7ba, #388edc);
        color: #041217;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 8px 28px rgba(39, 197, 190, 0.16);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }

    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #a0fff5;
        color: #031016;
        filter: brightness(1.08);
        box-shadow: 0 12px 34px rgba(39, 197, 190, 0.25);
    }

    [data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 2px #070b14, 0 0 0 4px var(--cyan);
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink));
    }

    [data-testid="stAlert"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(15, 26, 42, 0.84);
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: var(--cyan);
        box-shadow: 0 0 12px rgba(84, 242, 227, 0.35);
    }

    [data-testid="stDivider"] {
        border-color: var(--line);
    }

    .tech-card {
        height: 100%;
        padding: 1.25rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(16, 29, 47, 0.9), rgba(10, 16, 29, 0.88));
    }

    .tech-index {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }

    .tech-title {
        margin: 0.55rem 0;
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .tech-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.35rem 1rem 2rem;
        }

        .hero-shell {
            border-radius: 16px;
        }

        .hero-shell::after {
            right: -14rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def default_advanced_artifact_dir() -> Path:
    return project_root() / "outputs" / "advanced"


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
advanced_artifact_dir = default_advanced_artifact_dir()
advanced_available = (advanced_artifact_dir / "model_bundle.joblib").is_file()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">News<span>Check</span></div>
            <div class="sidebar-tagline">Credibility pattern analyzer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    model_options = ["Baseline · TF-IDF + Logistic Regression"]
    if advanced_available:
        model_options.append("Advanced · Local ML ensemble")
    selected_model = st.selectbox("Active classifier", model_options)
    use_advanced_model = selected_model.startswith("Advanced")

    if not advanced_available:
        st.caption(
            "Advanced local models are not trained yet. "
            "Run `python src/train_advanced.py` to create them."
        )

    st.divider()

    st.subheader("System Details")

    st.write("**Feature Extraction**")
    st.write("Transformer + sentence embeddings" if use_advanced_model else "TF-IDF")

    st.write("**Classifier**")
    st.write("Calibrated local ensemble" if use_advanced_model else "Logistic Regression")

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

st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-kicker">News intelligence / 01</div>
        <div class="hero-title">Read the <span>signal.</span><br>Question the noise.</div>
        <p class="hero-copy">
            Analyze the language patterns in a headline or article excerpt.
            NewsCheck estimates whether the text resembles examples from its
            real or fake training categories.
        </p>
        <div class="hero-status"><span class="status-dot"></span> Pattern analysis system</div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CHECK
# ============================================================

if not use_advanced_model and not pipeline_path.exists():

    st.error(
        "The trained classification model is unavailable."
    )

    st.info(
        "Train the model first with "
        "`python src/train_model.py`."
    )

    st.stop()


pipeline = None
if not use_advanced_model:
    pipeline = load_pipeline(str(pipeline_path))


# ============================================================
# NEWS INPUT
# ============================================================

st.markdown('<div class="section-kicker">01 / Submit a sample</div>', unsafe_allow_html=True)
st.subheader("News text")

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
    "Run signal analysis",
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
        if use_advanced_model:
            from advanced_models import predict_advanced

            advanced_result = predict_advanced(
                advanced_artifact_dir,
                text,
                threshold=threshold,
                uncertainty_margin=uncertainty_margin,
            )
            fake_probability = float(advanced_result["prob_fake"])
            prediction = str(advanced_result["label"])
        else:
            if pipeline is None:
                raise RuntimeError("The baseline classifier was not loaded.")
            fake_probability = float(pipeline.predict_proba([text])[0, 1])
            prediction = classify_probability(
                fake_probability,
                threshold,
                uncertainty_margin,
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

    if use_advanced_model and "advanced_result" in locals():
        with st.expander("Model probability breakdown"):
            component_probabilities = advanced_result.get("components", {})
            if component_probabilities:
                for component, probability in component_probabilities.items():
                    st.metric(
                        str(component).replace("_", " ").title(),
                        f"{float(probability):.1%}",
                    )
            st.caption(
                "The ensemble combines locally-run model outputs; confidence "
                "does not verify whether claims are factually true."
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

    st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)
    st.subheader("Analysis result")

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

    elif prediction == "UNCERTAIN":

        st.warning(
            "❓ INCONCLUSIVE RESULT"
        )

        st.write(
            "The prediction is close to the classification "
            "boundary, so the system cannot make a clear "
            "classification."
        )

    else:
        st.warning(
            "◈ INSUFFICIENT CONFIDENCE"
        )

        st.write(
            "The advanced model's calibrated uncertainty estimate is too high "
            "to assign a reliable category. Consider providing more context."
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

    st.subheader("Probability distribution")

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

    st.subheader("Text statistics")

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

    st.subheader("Interpretation")

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
            "The prediction did not meet the confidence requirement for either category."
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

st.markdown('<div class="section-kicker">03 / Under the hood</div>', unsafe_allow_html=True)
st.subheader("How the system reads text")

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-index">MODULE 01</div>
            <div class="tech-title">Text representation</div>
            <p class="tech-copy">The submitted language is converted into numerical features the model can process.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tech2:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-index">MODULE 02</div>
            <div class="tech-title">TF-IDF weighting</div>
            <p class="tech-copy">Word and phrase importance is measured against the model's training collection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tech3:

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-index">MODULE 03</div>
            <div class="tech-title">Classification</div>
            <p class="tech-copy">Logistic Regression estimates the probability of each learned language category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NewsCheck / Educational machine-learning application / Not a factual verification service"
)
