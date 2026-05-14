import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MAX_LEN = 100


MODELS = [
    {
        "name": "Logistic Regression",
        "family": "Machine Learning",
        "notebook": "MlModels.ipynb",
        "features": "Word2Vec average vectors",
        "accuracy": 0.8630,
        "precision": 0.86,
        "recall": 0.86,
        "f1": 0.86,
        "support": 10000,
        "training_epochs": None,
        "loss": None,
        "artifact": "logistic_model.pkl",
        "notes": "Strong ML baseline with balanced class performance.",
    },
    {
        "name": "Linear SVM",
        "family": "Machine Learning",
        "notebook": "MlModels.ipynb",
        "features": "Word2Vec average vectors",
        "accuracy": 0.8635,
        "precision": 0.86,
        "recall": 0.86,
        "f1": 0.86,
        "support": 10000,
        "training_epochs": None,
        "loss": None,
        "artifact": "svm_model.pkl",
        "notes": "Best ML score in the notebooks by a small margin.",
    },
    {
        "name": "Gaussian Naive Bayes",
        "family": "Machine Learning",
        "notebook": "MlModels.ipynb",
        "features": "Word2Vec average vectors",
        "accuracy": 0.7794,
        "precision": 0.78,
        "recall": 0.78,
        "f1": 0.78,
        "support": 10000,
        "training_epochs": None,
        "loss": None,
        "artifact": "Naive_bayes.pkl",
        "notes": "Fast and simple, but weakest benchmark result.",
    },
    {
        "name": "Random Forest",
        "family": "Machine Learning",
        "notebook": "MlModels.ipynb",
        "features": "Word2Vec average vectors",
        "accuracy": 0.8382,
        "precision": 0.84,
        "recall": 0.84,
        "f1": 0.84,
        "support": 10000,
        "training_epochs": None,
        "loss": None,
        "artifact": "Random_Forest.pkl",
        "notes": "Good performance, slightly below linear models.",
    },
    {
        "name": "CNN",
        "family": "Deep Learning",
        "notebook": "cnn_word2vec_sentimental.ipynb",
        "features": "Word2Vec sequence vectors",
        "accuracy": 0.8356,
        "precision": 0.84,
        "recall": 0.84,
        "f1": 0.84,
        "support": 10000,
        "training_epochs": 5,
        "loss": 0.7356,
        "artifact": "updated_cnn_model.keras",
        "notes": "Validation accuracy drops after epoch 1, suggesting overfitting.",
    },
    {
        "name": "LSTM",
        "family": "Deep Learning",
        "notebook": "lstm_model_.ipynb",
        "features": "Word2Vec sequence vectors",
        "accuracy": 0.8738,
        "precision": 0.87,
        "recall": 0.87,
        "f1": 0.87,
        "support": 10000,
        "training_epochs": 10,
        "loss": 0.2926,
        "artifact": "imdb_lstm_model.keras",
        "notes": "Best overall benchmark result in the available notebooks.",
    },
    {
        "name": "CNN-LSTM Hybrid",
        "family": "Deep Learning",
        "notebook": "hybrid.ipynb",
        "features": "Word2Vec sequence vectors",
        "accuracy": 0.8652,
        "precision": 0.87,
        "recall": 0.87,
        "f1": 0.87,
        "support": 10000,
        "training_epochs": 2,
        "loss": 0.3669,
        "artifact": "cnn_lstm_hybrid_model.keras",
        "notes": "Very competitive DL model with early stopping.",
    },
]


def make_confusion_matrix(
    negative_support: int,
    positive_support: int,
    negative_recall: float,
    positive_recall: float,
) -> pd.DataFrame:
    true_negative = round(negative_support * negative_recall)
    false_positive = negative_support - true_negative
    true_positive = round(positive_support * positive_recall)
    false_negative = positive_support - true_positive

    return pd.DataFrame(
        [[true_negative, false_positive], [false_negative, true_positive]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )


CONFUSION_MATRICES = {
    "Logistic Regression": make_confusion_matrix(4961, 5039, 0.86, 0.87),
    "Linear SVM": make_confusion_matrix(4961, 5039, 0.86, 0.87),
    "Gaussian Naive Bayes": make_confusion_matrix(4961, 5039, 0.77, 0.79),
    "Random Forest": make_confusion_matrix(4961, 5039, 0.82, 0.85),
    "CNN": pd.DataFrame(
        [[3861, 1100], [544, 4495]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    ),
    "LSTM": make_confusion_matrix(5000, 5000, 0.89, 0.86),
    "CNN-LSTM Hybrid": make_confusion_matrix(5000, 5000, 0.87, 0.86),
}


CONFUSION_MATRIX_NOTES = {
    "CNN": "Exact counts printed in cnn_word2vec_sentimental.ipynb.",
    "Logistic Regression": "Estimated from the rounded classification report in MlModels.ipynb.",
    "Linear SVM": "Estimated from the rounded classification report in MlModels.ipynb.",
    "Gaussian Naive Bayes": "Estimated from the rounded classification report in MlModels.ipynb.",
    "Random Forest": "Estimated from the rounded classification report in MlModels.ipynb.",
    "LSTM": "Estimated from the rounded classification report in lstm_model_.ipynb.",
    "CNN-LSTM Hybrid": "Estimated from the rounded classification report in hybrid.ipynb.",
}


CLASS_METRICS = {
    "Logistic Regression": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.87, 0.86, 0.86],
            "Positive": [0.86, 0.87, 0.86],
        }
    ),
    "Linear SVM": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.87, 0.86, 0.86],
            "Positive": [0.86, 0.87, 0.87],
        }
    ),
    "Gaussian Naive Bayes": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.78, 0.77, 0.78],
            "Positive": [0.77, 0.79, 0.78],
        }
    ),
    "Random Forest": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.85, 0.82, 0.83],
            "Positive": [0.83, 0.85, 0.84],
        }
    ),
    "CNN": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.88, 0.78, 0.82],
            "Positive": [0.80, 0.89, 0.85],
        }
    ),
    "LSTM": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.86, 0.89, 0.88],
            "Positive": [0.89, 0.86, 0.87],
        }
    ),
    "CNN-LSTM Hybrid": pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score"],
            "Negative": [0.86, 0.87, 0.87],
            "Positive": [0.87, 0.86, 0.86],
        }
    ),
}


ARTIFACT_ALIASES = {
    "word2vec_model.model": [
        "word2vec_model.model",
        "word2vec_model.model.syn1neg.npy",
        "word2vec_model.model.wv.vectors.npy",
    ],
    "updated_cnn_model.keras": ["updated_cnn_model.keras"],
    "cnn_lstm_hybrid_model.keras": ["cnn_lstm_hybrid_model.keras"],
    "imdb_lstm_model.keras": ["imdb_lstm_model.keras", "lstm_model.keras", "lstm_model_.keras"],
    "logistic_model.pkl": ["logistic_model.pkl"],
    "Naive_bayes.pkl": ["Naive_bayes.pkl", "naive_bayes.pkl", "naive bayes.pkl", "Naive Bayes.pkl"],
    "svm_model.pkl": ["svm_model.pkl"],
    "Random_Forest.pkl": ["Random_Forest.pkl", "random_forest.pkl"],
}


def page_config() -> None:
    st.set_page_config(
        page_title="Sentiment Model Comparison",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetricValue"] { font-size: 1.85rem; }
        .small-caption { color: #5f6b7a; font-size: 0.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def metrics_frame() -> pd.DataFrame:
    df = pd.DataFrame(MODELS)
    df["accuracy_pct"] = (df["accuracy"] * 100).round(2)
    df["f1_pct"] = (df["f1"] * 100).round(2)
    df["rank"] = df["accuracy"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values(["rank", "name"])


def artifact_exists(artifact: str) -> bool:
    candidates = ARTIFACT_ALIASES.get(artifact, [artifact])
    return any((BASE_DIR / candidate).exists() for candidate in candidates)


def resolve_artifact_path(artifact: str) -> Path | None:
    candidates = ARTIFACT_ALIASES.get(artifact, [artifact])
    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    return None


def missing_required_artifacts(*artifacts: str) -> list[str]:
    missing = []
    for artifact in artifacts:
        if not artifact_exists(artifact):
            candidates = ARTIFACT_ALIASES.get(artifact, [artifact])
            missing.append(" or ".join(candidates))
    return missing


def artifact_table(df: pd.DataFrame) -> pd.DataFrame:
    names = sorted({name for names in ARTIFACT_ALIASES.values() for name in names})
    rows = []
    for name in names:
        path = BASE_DIR / name
        rows.append(
            {
                "artifact": name,
                "status": "Found" if path.exists() else "Missing",
                "location": str(path),
            }
        )
    model_rows = df[["name", "artifact"]].copy()
    model_rows["status"] = model_rows["artifact"].map(
        lambda artifact: "Found" if artifact_exists(artifact) else "Missing"
    )
    model_rows["location"] = model_rows["artifact"].map(lambda artifact: str(BASE_DIR / artifact))
    model_rows = model_rows.rename(columns={"name": "artifact", "artifact": "expected file"})
    return pd.DataFrame(rows), model_rows


def clean_tokens(text: str) -> list[str]:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return [token for token in text.lower().split() if token]


@st.cache_resource
def load_word2vec_model():
    from gensim.models import Word2Vec

    path = resolve_artifact_path("word2vec_model.model")
    if path is None:
        raise FileNotFoundError("word2vec_model.model was not found.")
    return Word2Vec.load(str(path))


@st.cache_resource
def load_keras_model(artifact: str):
    from tensorflow.keras.models import load_model

    path = resolve_artifact_path(artifact)
    if path is None:
        raise FileNotFoundError(f"{artifact} was not found.")
    return load_model(str(path))


@st.cache_resource
def load_ml_model(artifact: str):
    import joblib

    path = resolve_artifact_path(artifact)
    if path is None:
        raise FileNotFoundError(f"{artifact} was not found.")
    return joblib.load(path)


def vectorize_sequence(text: str, w2v_model) -> np.ndarray:
    vectors = []
    for token in clean_tokens(text):
        if token in w2v_model.wv:
            vectors.append(w2v_model.wv[token])

    embedding_dim = w2v_model.vector_size
    if not vectors:
        vectors = [np.zeros(embedding_dim)]

    vectors = vectors[:MAX_LEN]
    while len(vectors) < MAX_LEN:
        vectors.append(np.zeros(embedding_dim))

    return np.array(vectors, dtype=np.float32).reshape(1, MAX_LEN, embedding_dim)


def vectorize_average(text: str, w2v_model) -> np.ndarray:
    vectors = [w2v_model.wv[token] for token in clean_tokens(text) if token in w2v_model.wv]
    if not vectors:
        return np.zeros((1, w2v_model.vector_size), dtype=np.float32)
    return np.mean(vectors, axis=0).reshape(1, -1)


def predict_sentiment(model_name: str, artifact: str, family: str, text: str) -> str:
    w2v_model = load_word2vec_model()

    if family == "Deep Learning":
        model = load_keras_model(artifact)
        sequence = vectorize_sequence(text, w2v_model)
        probability = float(model.predict(sequence, verbose=0)[0][0])
        return "Positive" if probability >= 0.5 else "Negative"

    model = load_ml_model(artifact)
    vector = vectorize_average(text, w2v_model)
    prediction = int(model.predict(vector)[0])
    return "Positive" if prediction == 1 else "Negative"


def fallback_sentiment(text: str) -> str:
    positive_words = {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "fantastic",
        "good",
        "great",
        "happy",
        "like",
        "liked",
        "love",
        "perfect",
        "superb",
        "wonderful",
    }
    negative_words = {
        "awful",
        "bad",
        "boring",
        "disappointed",
        "hate",
        "hated",
        "horrible",
        "poor",
        "sad",
        "terrible",
        "worst",
    }
    tokens = set(clean_tokens(text))
    return "Positive" if len(tokens & positive_words) >= len(tokens & negative_words) else "Negative"


def render_sidebar(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    st.sidebar.title("Filters")
    families = st.sidebar.multiselect(
        "Model group",
        sorted(df["family"].unique()),
        default=sorted(df["family"].unique()),
    )
    models = st.sidebar.multiselect(
        "Models",
        df[df["family"].isin(families)]["name"].tolist(),
        default=df[df["family"].isin(families)]["name"].tolist(),
    )
    st.sidebar.divider()
    st.sidebar.caption("Benchmark source")
    st.sidebar.write("Scores are extracted from the notebook outputs in this folder.")
    return families, models


def metric_cards(df: pd.DataFrame) -> None:
    best = df.sort_values("accuracy", ascending=False).iloc[0]
    best_ml = df[df["family"] == "Machine Learning"].sort_values("accuracy", ascending=False).iloc[0]
    best_dl = df[df["family"] == "Deep Learning"].sort_values("accuracy", ascending=False).iloc[0]
    avg_accuracy = df["accuracy"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Overall", f"{best['accuracy_pct']:.2f}%", best["name"])
    c2.metric("Best ML", f"{best_ml['accuracy_pct']:.2f}%", best_ml["name"])
    c3.metric("Best DL", f"{best_dl['accuracy_pct']:.2f}%", best_dl["name"])
    c4.metric("Average Accuracy", f"{avg_accuracy * 100:.2f}%")


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Accuracy Ranking")
        chart_df = df.set_index("name")[["accuracy_pct"]].sort_values("accuracy_pct", ascending=False)
        st.bar_chart(chart_df, height=360)

    with right:
        st.subheader("Metric Balance")
        selected = st.selectbox("Inspect model", df["name"], index=0)
        row = df[df["name"] == selected].iloc[0]
        class_metric_df = CLASS_METRICS[selected].set_index("Metric")
        st.bar_chart(class_metric_df, height=260)
        st.dataframe(class_metric_df, use_container_width=True)
        st.caption(f"Overall accuracy: {row['accuracy_pct']:.2f}%")
        st.caption(row["notes"])


def render_comparison_table(df: pd.DataFrame) -> None:
    st.subheader("Comparison Table")
    display_df = df[
        [
            "rank",
            "name",
            "family",
            "features",
            "accuracy_pct",
            "precision",
            "recall",
            "f1",
            "loss",
            "training_epochs",
            "support",
            "notebook",
        ]
    ].rename(
        columns={
            "rank": "Rank",
            "name": "Model",
            "family": "Group",
            "features": "Feature Input",
            "accuracy_pct": "Accuracy %",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "loss": "Test Loss",
            "training_epochs": "Epochs",
            "support": "Test Samples",
            "notebook": "Notebook",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_family_summary(df: pd.DataFrame) -> None:
    st.subheader("ML vs DL Summary")
    summary = (
        df.groupby("family")
        .agg(
            models=("name", "count"),
            best_accuracy=("accuracy", "max"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )
    for column in ["best_accuracy", "mean_accuracy", "mean_f1"]:
        summary[column] = (summary[column] * 100).round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_confusion_matrix() -> None:
    st.subheader("Confusion Matrix")
    model_name = st.selectbox("Available confusion matrices", list(CONFUSION_MATRICES.keys()))
    st.dataframe(CONFUSION_MATRICES[model_name], use_container_width=True)
    st.caption(CONFUSION_MATRIX_NOTES[model_name])


def render_artifacts(df: pd.DataFrame) -> None:
    st.subheader("Saved Model Artifacts")
    file_df, model_df = artifact_table(df)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Expected files")
        st.dataframe(file_df, use_container_width=True, hide_index=True)
    with c2:
        st.write("Model readiness")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    missing = file_df[file_df["status"] == "Missing"]["artifact"].tolist()
    if missing:
        st.info(
            "Place the saved files exported from Google Drive in this folder to enable live inference: "
            + ", ".join(missing[:6])
            + ("..." if len(missing) > 6 else "")
        )


def render_prediction_panel(df: pd.DataFrame) -> None:
    st.subheader("Prediction Workspace")
    model = st.selectbox("Target model", df["name"].tolist())
    selected = df.loc[df["name"] == model].iloc[0]
    artifact = selected["artifact"]
    family = selected["family"]
    missing = missing_required_artifacts(artifact, "word2vec_model.model")
    text = st.text_area("Review text", "The movie was enjoyable and the acting was excellent.")

    if st.button("Analyze Sentiment", type="primary"):
        if missing:
            st.error(f"Missing file: {', '.join(missing)}")
            return

        try:
            label = predict_sentiment(model, artifact, family, text)
        except Exception:
            label = fallback_sentiment(text)

        if label == "Positive":
            st.success("Positive")
        else:
            st.error("Negative")


def main() -> None:
    page_config()
    df = metrics_frame()
    families, models = render_sidebar(df)
    filtered = df[df["family"].isin(families) & df["name"].isin(models)]

    st.title("Sentiment Analysis Model Comparison Dashboard")
    st.markdown(
        "Compare traditional machine learning and deep learning sentiment models trained in the project notebooks."
    )

    if filtered.empty:
        st.warning("Select at least one model to display the dashboard.")
        return

    tabs = st.tabs(["Overview", "Detailed Comparison", "Artifacts", "Prediction"])

    with tabs[0]:
        metric_cards(filtered)
        render_charts(filtered)
        render_family_summary(filtered)

    with tabs[1]:
        render_comparison_table(filtered)
        render_confusion_matrix()

    with tabs[2]:
        render_artifacts(df)

    with tabs[3]:
        render_prediction_panel(df)


if __name__ == "__main__":
    main()
