import json
from pathlib import Path
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

DATA = Path("data/test.jsonl")
MODEL_DIR = Path("models")
EMBED_MODEL = "intfloat/e5-small-v2"

UTIL = {
    "ANSWER_PREFER": {"route_answer": +2, "abstain": -1},
    "ABSTAIN_PREFER": {"route_answer": -1, "abstain": +2},
    "ABSTAIN": {"route_answer": -3, "abstain": +2},
}

def calibrate_probs(raw, cal_models):
    return np.stack([cal_models[c].transform(raw[:, c]) for c in range(raw.shape[1])], axis=1)

def main():
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    meta = load(MODEL_DIR / "meta.joblib")
    clf = load(MODEL_DIR / "clf.joblib")
    calib = load(MODEL_DIR / "calib.joblib")
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    le = meta["label_encoder"]
    class_names = list(le.classes_)
    idx = {name: i for i, name in enumerate(class_names)}

    emb = SentenceTransformer(EMBED_MODEL)
    X = emb.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    raw = clf.predict_proba(X)
    prob = calibrate_probs(raw, calib["cal_models"])

    util = 0.0; abstain = 0; answer = 0
    for n in range(len(texts)):
        pA = prob[n, idx.get("ANSWER_PREFER", -1)] if "ANSWER_PREFER" in idx else 0.0
        pS = prob[n, idx.get("ABSTAIN_PREFER", -1)] if "ABSTAIN_PREFER" in idx else 0.0

        if pA >= thresholds["tau_answer"]:
            action = "route_answer"; answer += 1
        elif pS >= thresholds["tau_abstainpref"]:
            action = "abstain"; abstain += 1
        else:
            action = "abstain"; abstain += 1

        util += UTIL[labels[n]][action]

    N = len(texts)
    print(json.dumps({
        "N": N,
        "expected_utility": util,
        "abstain_rate": abstain / N,
        "answer_rate": answer / N
    }, indent=2))

if __name__ == "__main__":
    main()
