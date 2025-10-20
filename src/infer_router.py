# src/infer_router.py
import sys, json
from pathlib import Path
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path("models")
EMBED_MODEL = "intfloat/multilingual-e5-small"

# --- add near the top, with thresholds ---
MARGIN_DELTA = 0.10  # how much ABSTAIN_PREFER must beat ANSWER_PREFER to force abstain-prefer

def calibrate_probs(raw, cal_models):
    # raw: [N, C] -> calibrated: [N, C]
    return np.stack([cal_models[c].transform(raw[:, c]) for c in range(raw.shape[1])], axis=1)

def decide(calib_prob, class_names, thresholds):
    idx = {name: i for i, name in enumerate(class_names)}
    pA = calib_prob[0, idx.get("ANSWER_PREFER", -1)] if "ANSWER_PREFER" in idx else 0.0
    pS = calib_prob[0, idx.get("ABSTAIN_PREFER", -1)] if "ABSTAIN_PREFER" in idx else 0.0

    # 1) If answer-prob is strong enough, answer.
    if pA >= thresholds["tau_answer"]:
        return "ANSWER_PREFER", {"p_answer_prefer": float(pA), "p_abstain_prefer": float(pS)}

    # 2) Only abstain-prefer if it CLEARLY beats answer by a margin
    if (pS - pA) >= MARGIN_DELTA and pS >= thresholds["tau_abstainpref"]:
        return "ABSTAIN_PREFER", {"p_answer_prefer": float(pA), "p_abstain_prefer": float(pS)}

    # 3) Otherwise play it safe
    return "ABSTAIN", {"p_answer_prefer": float(pA), "p_abstain_prefer": float(pS)}

def main():
    # Load artifacts
    meta = load(MODEL_DIR / "meta.joblib")
    clf = load(MODEL_DIR / "clf.joblib")
    calib = load(MODEL_DIR / "calib.joblib")
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    le = meta["label_encoder"]

    # Read input
    text = sys.stdin.read().strip() if not sys.argv[1:] else " ".join(sys.argv[1:])
    if not text:
        print(json.dumps({"decision": "ABSTAIN", "probs": {"p_answer_prefer": 0.0, "p_abstain_prefer": 0.0}}))
        return

    # Embed + predict
    embedder = SentenceTransformer(EMBED_MODEL)
    X = embedder.encode([text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    raw = clf.predict_proba(X)
    prob = calibrate_probs(raw, calib["cal_models"])

    # Decide
    decision, probs = decide(prob, list(le.classes_), thresholds)
    print(json.dumps({"decision": decision, "probs": probs}))

if __name__ == "__main__":
    main()
