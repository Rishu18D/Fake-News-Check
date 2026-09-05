# Data Statement

## Dataset Role

The dataset used by **NewsCheck** is intended for educational binary text
classification.

It is used to demonstrate how a machine-learning model can learn patterns from
news text and classify examples into `REAL` and `FAKE` categories.

The dataset is suitable for:

- Learning NLP concepts
- Training a text-classification model
- Experimenting with TF-IDF
- Evaluating Logistic Regression
- Demonstrating a Streamlit machine-learning application

It should not be considered sufficient for real-world misinformation or
fact-checking applications.

---

## Dataset Files

The project expects the following files:

- `data/True.csv` — examples labeled as REAL
- `data/Fake.csv` — examples labeled as FAKE

The training process uses the available news-text fields from these datasets
to create the input used by the machine-learning pipeline.

---

## Dataset Usage

The dataset is processed before model training and is then divided into
training and evaluation data.

The model learns statistical relationships between the text and the labels.

The resulting model should therefore be understood as a **text-pattern
classifier**, rather than a factual verification system.

---

## Dataset Limitations

The quality of a machine-learning classifier depends heavily on the quality and
diversity of its training data.

Potential limitations include:

- Limited dataset size
- Class imbalance
- Repeated or duplicated articles
- Source-specific writing styles
- Topic-specific vocabulary
- Historical differences in news writing
- Publisher-specific patterns
- Possible data leakage
- Differences between training data and modern news

These limitations can cause the model to perform differently on unseen news
sources or topics.

---

## Bias and Leakage Considerations

News datasets can contain patterns that allow a classifier to distinguish
between classes without actually understanding whether a claim is true.

For example, the model may learn:

- Publisher-specific writing styles
- Repeated phrases
- Formatting patterns
- Article templates
- Source names
- Image-credit text
- Topic-specific vocabulary
- Other dataset-specific artifacts

Therefore, strong evaluation results should not automatically be interpreted
as evidence of reliable real-world fake-news detection.

---

## Recommended Improvements

Before using NewsCheck beyond an educational demonstration, the dataset should
ideally be improved by:

1. Increasing the size and diversity of the dataset.
2. Balancing examples across multiple publishers.
3. Removing duplicate articles.
4. Reducing source-specific artifacts.
5. Using time-separated evaluation data.
6. Testing on publishers not present in the training data.
7. Evaluating performance across different topics and sources.
8. Monitoring model performance on newer news articles.
9. Adding external evidence verification.
10. Keeping human review for important decisions.

---

## Responsible Dataset Use

The dataset should be used only in accordance with its original licensing and
usage requirements.

Users should verify the licensing terms of any dataset or external data added
to this project before redistribution or commercial use.

The labels in the dataset should also not be treated as absolute proof that an
article is factually true or false.

---

## Summary

The dataset provides a useful foundation for demonstrating an educational
NLP classification workflow.

However, NewsCheck's predictions are limited by the information and patterns
contained in the training data.

For this reason, the system should be used as a **learning and pattern-analysis
tool**, not as an automated fact-checker.
