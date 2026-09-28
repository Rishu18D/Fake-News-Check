from sklearn.pipeline import Pipeline

from train_model import build_labeled_frame, build_pipeline


def test_pipeline_can_fit_and_predict_small_sample():
    pipeline = build_pipeline(max_features=100, min_df=1, max_df=1.0, ngram_max=1, C=1.0)
    X = [
        "reuters official policy statement government economy",
        "reuters senate committee announces new bill",
        "shocking celebrity secret exposed fake claim",
        "viral hoax says impossible miracle happened",
    ]
    y = [0, 0, 1, 1]
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(["reuters government statement"])[0]
    assert isinstance(pipeline, Pipeline)
    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) < 1e-9


def test_conflicting_cross_label_duplicates_are_removed(tmp_path):
    real_path = tmp_path / "real.csv"
    fake_path = tmp_path / "fake.csv"
    real_path.write_text("text\nsame story\nreal story\n", encoding="utf-8")
    fake_path.write_text("text\nsame story\nfake story\n", encoding="utf-8")

    data, profile = build_labeled_frame(real_path, fake_path, "text", include_title=False)

    assert "same story" not in data["text_for_model"].tolist()
    assert profile.conflicting_duplicate_rows_removed == 2
    assert profile.rows_after_deduplication == 2
