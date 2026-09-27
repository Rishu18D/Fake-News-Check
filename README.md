<div align="center">

# 🛡️ NewsCheck

### News Credibility Pattern Analyzer

A machine-learning application that analyzes news text and predicts whether its language is more closely associated with **REAL** or **FAKE** news patterns.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20Logistic%20Regression-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![Status](https://img.shields.io/badge/Status-Educational%20Project-purple)

</div>

---

## 📌 Project Overview

**NewsCheck** is an educational machine-learning project designed to analyze news headlines and article excerpts.

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features and **Logistic Regression** to classify the input into:

- `REAL`
- `FAKE`
- `UNCERTAIN`

The project provides an interactive **Streamlit web application** where users can enter news text and receive a prediction along with probability scores and model confidence.

> ⚠️ **Important:** NewsCheck is a machine-learning classification system, not a professional fact-checking system. It does not independently verify facts, sources, dates, people, or events.

---

## 🎯 What NewsCheck Does

NewsCheck can:

- Analyze a news headline or article excerpt
- Process natural-language news text
- Extract TF-IDF features from the input
- Use Logistic Regression for classification
- Estimate the probability of the FAKE category
- Estimate the probability of the REAL category
- Return `REAL`, `FAKE`, or `UNCERTAIN`
- Display model confidence
- Show text statistics
- Provide an adjustable classification threshold
- Provide an uncertainty margin
- Display model evaluation metrics
- Run through an interactive Streamlit dashboard

---

## 🚫 What NewsCheck Does Not Do

NewsCheck does **not**:

- Verify whether a news story is factually true
- Search Google or other search engines for evidence
- Verify sources or journalists
- Check claims against government databases
- Guarantee that a prediction is correct
- Replace professional fact-checkers
- Detect every form of misinformation
- Make high-stakes decisions

The prediction represents how closely the language resembles patterns learned from the training dataset.

---

## 🧠 How It Works

The basic workflow is:

```text
User enters news text
        ↓
Text preprocessing
        ↓
TF-IDF feature extraction
        ↓
Logistic Regression
        ↓
Fake-news probability
        ↓
Classification logic
        ↓
REAL / FAKE / UNCERTAIN
        ↓
Result + Probability + Confidence

## Advanced local ML mode

The original TF-IDF + Logistic Regression pipeline remains the lightweight
baseline. An optional advanced training workflow adds a locally-run transformer,
Sentence-Transformer semantic features, attention-based feature fusion, and a
validation-calibrated ensemble. The dashboard exposes the advanced ensemble
after its artifacts have been trained.

Install the optional open-source dependencies in the same Python environment:

```powershell
python -m pip install -r requirements-advanced.txt
```

Train the additional models without replacing the baseline artifacts:

```powershell
python src/train_advanced.py --real data/True.csv --fake data/Fake.csv --outdir outputs/advanced
```

Model weights are obtained from their public Hugging Face model repositories
when training runs for the first time, then saved under `outputs/advanced` for
local inference. No API service, API key, RAG, or web search is used. Training
and inference are local; CPU is supported, while CUDA is used automatically
when available. Transformer training can take substantial time and disk space
on CPU. The transformer and sentence-embedding checkpoints can be changed with
the training command's model options. CPU-friendly defaults cap neural
training at 1,000 stratified training rows and evaluation at 300 stratified
validation/test rows; the baseline still trains on its full training split.
Use `--neural-train-limit` and `--neural-eval-limit` to change these limits.
Comparisons therefore describe the configured training protocol, not a
same-compute or same-training-volume benchmark.

The advanced evaluation writes model comparisons, validation protocol details,
and ablation results to `outputs/advanced/advanced_metrics.json`. It keeps the
holdout test set separate from training, model selection, and probability
calibration. Integrated Gradients token attributions are available for the
transformer through the advanced explainability API. Treat the report
cautiously: the bundled dataset has known source
and writing-style artifacts, and good benchmark metrics do not mean that a
model can verify claims or generalize to current news. Advanced uncertainty
may return **Insufficient Confidence** rather than forcing a prediction.

To use the baseline instead, leave `outputs/advanced` untrained or choose the
baseline classifier in the dashboard. Its CLI and existing model artifacts
continue to work as before.
