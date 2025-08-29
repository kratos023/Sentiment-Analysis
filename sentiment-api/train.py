import pandas as pd, joblib, re, string, nltk, os, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

DATA_PATH = "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\data\\twitter_training.csv"
VALIDATION_PATH = "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\data\\twitter_validation.csv"
MODEL_DIR = "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\model"
MLRUN_DIR = "C:\\Users\\Deepanshu\\Downloads\\sentiment-api\\mlruns1"

mlflow.set_tracking_uri(f"file:///{MLRUN_DIR}")
mlflow.set_experiment("sentiment-api")

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

def log_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    mlflow.log_figure(fig, filename)
    plt.close(fig)

def log_metrics(prefix, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ✅ log both prefix metrics and top-level
    mlflow.log_metric(f"{prefix}_accuracy", acc)
    mlflow.log_metric(f"{prefix}_precision", prec)
    mlflow.log_metric(f"{prefix}_recall", rec)
    mlflow.log_metric(f"{prefix}_f1", f1)

    if prefix == "test":  # ✅ also log top-level so UI shows graphs
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

# ─────────────────────────────────────────────
# Load and preprocess training data
# ─────────────────────────────────────────────
cols = ["ID", "Entity", "Sentiment", "Tweet"]
df = pd.read_csv(DATA_PATH, names=cols).dropna(subset=["Tweet"]).drop_duplicates()
df = df[df["Sentiment"].isin(["Positive", "Negative"])]
df["clean"] = df["Tweet"].apply(clean)

le = LabelEncoder()
df["y"] = le.fit_transform(df["Sentiment"])

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
)

# ─────────────────────────────────────────────
# MLflow run
# ─────────────────────────────────────────────
with mlflow.start_run() as run:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=300, n_jobs=-1, C=4.0)),
    ])
    pipe.fit(X_train, y_train)

    # Params
    mlflow.log_params({
        "model": "LogisticRegression",
        "ngram_range": "(1,2)",
        "min_df": 3,
        "C": 4.0,
        "max_iter": 300
    })

    # ───── Test Metrics ─────
    preds = pipe.predict(X_test)
    log_metrics("test", y_test, preds)
    log_confusion_matrix(y_test, preds, le.classes_, "Confusion Matrix - Test", "confusion_matrix_test.png")

    # Save classification report ✅
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
    with open("test_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("test_classification_report.json")

    # ───── Validation Metrics ─────
    val_df = pd.read_csv(VALIDATION_PATH, names=cols).dropna(subset=["Tweet"])
    val_df = val_df[val_df["Sentiment"].isin(["Positive", "Negative"])]
    val_df["clean"] = val_df["Tweet"].apply(clean)
    val_y_true = le.transform(val_df["Sentiment"])
    val_preds = pipe.predict(val_df["clean"])
    log_metrics("val", val_y_true, val_preds)
    log_confusion_matrix(val_y_true, val_preds, le.classes_, "Confusion Matrix - Validation", "confusion_matrix_val.png")

    # ───── Log model ─────
    input_example = pd.DataFrame({"clean": [X_test.iloc[0]]})   # ✅ single row
    signature = infer_signature(pd.DataFrame({"clean": X_test}), preds)  # ✅ fixed signature

    mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Save locally too
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, f"{MODEL_DIR}\\model.joblib")
    joblib.dump(le, f"{MODEL_DIR}\\label_encoder.joblib")

    # Print summary
    print("✅ Test Accuracy:", accuracy_score(y_test, preds))
    print("✅ Validation Accuracy:", accuracy_score(val_y_true, val_preds))
    print("🧾 Test Report:\n", classification_report(y_test, preds, target_names=le.classes_))
    print("🧾 Validation Report:\n", classification_report(val_y_true, val_preds, target_names=le.classes_))

# Explicitly end run
mlflow.end_run()
