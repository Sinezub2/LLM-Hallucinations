import json
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL = "intfloat/multilingual-e5-small"
   # compact, high-quality encoder

def load_split(name):
    path = DATA_DIR / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]
    return X, y

def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    # Encode labels
    le = LabelEncoder().fit(y_train + y_val)
    ytr = le.transform(y_train)
    yva = le.transform(y_val)

    # Embed text
    model = SentenceTransformer(EMBED_MODEL)
    Xtr = model.encode(X_train, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    Xva = model.encode(X_val, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto")
    clf.fit(Xtr, ytr)

    # Evaluate on val
    y_pred = clf.predict(Xva)
    print(classification_report(yva, y_pred, target_names=le.classes_))

    # Save artifacts
    dump({"label_encoder": le}, MODEL_DIR / "meta.joblib")
    dump(clf, MODEL_DIR / "clf.joblib")
    # We'll re-instantiate the embedder on load; no need to pickle full model here
    print("Saved models/meta.joblib and clf.joblib")

if __name__ == "__main__":
    main()
