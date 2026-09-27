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
Likely Credible / Likely Misleading / Uncertain / Insufficient Confidence
        ↓
Result + Probability + Confidence

## Advanced local ML mode

The original TF-IDF + Logistic Regression pipeline remains the lightweight
baseline. An optional advanced training workflow adds a locally-run transformer,
Sentence-Transformer semantic features, attention-based feature fusion, and a
validation-calibrated ensemble. Trained weights are local artifacts under
`outputs/advanced` and are intentionally excluded from source control; the
dashboard offers only components whose required local files are present.
When an ensemble is ready, it is selected by default while retaining the
baseline as an option.

Start the application from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\streamlit_app.py
```

The saved Transformer, sentence embedding, and fusion artifacts require the
optional runtime dependencies below. The baseline classifier remains usable
without them.

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

The local development `outputs/advanced` bundle was trained on CPU with a
96-example neural training cap, a 40-example validation/test cap, batch size 8,
and one epoch so the Transformer, embeddings, fusion model, and ensemble are
available in the development workspace. The artifact directory is ignored by
Git and is not included in a source checkout. To generate it locally, use the
training command above. These deliberately small caps are a smoke-trained local
model, not a production-quality benchmark. For a more representative run, use
the default 1,000/300 caps (or larger limits) and review the new holdout report.

The advanced evaluation writes model comparisons, validation protocol details,
and ablation results to `outputs/advanced/advanced_metrics.json`. It keeps the
holdout test set separate from training, model selection, and probability
calibration. Component calibrators use validation predictions; ensemble
calibration uses validation-only out-of-fold predictions. The report compares
raw and Platt-scaled holdout probabilities using accuracy, precision, recall,
F1, ROC-AUC, PR-AUC, Brier score, confusion matrices, and calibration curves.
Integrated Gradients token attributions are available from the dashboard for
the Transformer and through the advanced explainability API. The dashboard's
evidence-style card summarizes the analyzed text, prediction, confidence, and
model indicators; it is explicitly not verified external evidence. Set the
minimum confidence threshold in the dashboard to control when predictions
return **Insufficient Confidence**. Treat metrics cautiously: the bundled
dataset has known source and writing-style artifacts, and good benchmark
metrics do not mean that a model can verify claims or generalize to current
news.

To use the baseline instead, leave `outputs/advanced` untrained or choose the
baseline classifier in the dashboard. Its CLI and existing model artifacts
continue to work as before.
