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
