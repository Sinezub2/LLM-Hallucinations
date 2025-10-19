import json
from pathlib import Path
import numpy as np
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def load_split(name):
    rows = [json.loads(l) for l in Path(DATA_DIR, f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]
    return X, y

def main():
    meta = load(MODEL_DIR / "meta.joblib")
    clf = load(MODEL_DIR / "clf.joblib")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("intfloat/e5-small-v2")

    X_val, y_val = load_split("val")
    Xv = embedder.encode(X_val, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    # Raw probabilities
    raw = clf.predict_proba(Xv)
    labels = meta["label_encoder"].transform(y_val)

    cal_models = []
    for c in range(raw.shape[1]):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw[:, c], (labels == c).astype(float))
        cal_models.append(ir)

    dump({"cal_models": cal_models}, MODEL_DIR / "calib.joblib")
    print("Saved calibration to models/calib.joblib")

if __name__ == "__main__":
    main()
