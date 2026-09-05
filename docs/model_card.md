# Model Card: NewsCheck

## Model Overview

**NewsCheck** is an educational machine-learning application that classifies
news text according to patterns learned from a labeled training dataset.

### Model Details

- **Project:** NewsCheck
- **Task:** Binary text classification
- **Output:** `REAL`, `FAKE`, or `UNCERTAIN`
- **Feature Extraction:** TF-IDF
- **Classifier:** Logistic Regression
- **Framework:** scikit-learn
- **Application:** Streamlit
- **Model Artifact:** `outputs/pipeline.joblib`

---

## Intended Use

NewsCheck is intended for:

- Educational machine-learning demonstrations
- NLP learning
- Text-classification experiments
- Portfolio projects
- Demonstrating TF-IDF and Logistic Regression
- Exploring probability-based classification
- Demonstrating uncertainty handling
- Interactive Streamlit applications

---

## Out-of-Scope Uses

NewsCheck should not be used as the sole basis for:

- Fact-checking
- News censorship
- Content moderation
- Political decision-making
- Reputation decisions
- Legal decisions
- Employment decisions
- Financial decisions
- Any other high-stakes decision

The model does not independently establish whether a claim is true or false.

---

## Model Architecture

The basic model pipeline is:

```text
News Text
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Fake Probability
    ↓
Classification Decision
    ↓
REAL / FAKE / UNCERTAIN
