# src/pick_thresholds.py
import json
from pathlib import Path
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# ---- Utility table (tune to your preference) ----
UTIL = {
    "ANSWER_PREFER": {"route_answer": +2, "abstain": -1},
    "ABSTAIN_PREFER": {"route_answer": -1, "abstain": +2},
    "ABSTAIN": {"route_answer": -3, "abstain": +2},
}

# ---- Search/constraint settings ----
EMBED_MODEL = "intfloat/multilingual-e5-small"

GRID = {
    "tau_answer":  np.linspace(0.50, 0.95, 10),
    "tau_abstain": np.linspace(0.50, 0.95, 10),
}
TARGETS = {
    "abstain_rate_max": 0.40,  # cap abstentions (e.g., 40%)
    "min_answer_rate":  0.30,  # ensure we answer at least this often
}
TOP_K = 5  # return top-K candidates by expected utility (under constraints)

def load_split(name):
    rows = [json.loads(l) for l in Path(DATA_DIR, f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]
    return X, y

def calibrate_probs(raw, cal_models):
    # raw: [N, C] -> calibrated probs: [N, C]
    return np.stack([cal_models[c].transform(raw[:, c]) for c in range(raw.shape[1])], axis=1)

def main():
    # Load artifacts
    meta = load(MODEL_DIR / "meta.joblib")
    clf = load(MODEL_DIR / "clf.joblib")
    calib = load(MODEL_DIR / "calib.joblib")
    le = meta["label_encoder"]
    class_names = list(le.classes_)
    idx = {name: i for i, name in enumerate(class_names)}

    # Data
    X_val, y_val = load_split("val")
    y_val_i = le.transform(y_val)

    # Embed + raw probs
    embedder = SentenceTransformer(EMBED_MODEL)
    Xv = embedder.encode(X_val, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    raw = clf.predict_proba(Xv)
    prob = calibrate_probs(raw, calib["cal_models"])

    # Unconstrained best (for fallback)
    best_unconstrained = {"tau_answer": 0.7, "tau_abstainpref": 0.8, "utility": -1e9, "metrics": {}}

    candidates = []
    for ta in GRID["tau_answer"]:
        for ts in GRID["tau_abstain"]:
            util = 0.0
            abstain_cnt = 0
            answer_cnt = 0

            for n in range(prob.shape[0]):
                pA = prob[n, idx.get("ANSWER_PREFER", -1)] if "ANSWER_PREFER" in idx else 0.0
                pS = prob[n, idx.get("ABSTAIN_PREFER", -1)] if "ABSTAIN_PREFER" in idx else 0.0

                # Two-path policy:
                # If confident it's answer-prefer -> route answer
                # Else if confident it's abstain-prefer -> abstain
                # Else default to abstain (conservative)
                if pA >= ta:
                    action = "route_answer"; answer_cnt += 1
                elif pS >= ts:
                    action = "abstain"; abstain_cnt += 1
                else:
                    action = "abstain"; abstain_cnt += 1

                true_class = class_names[y_val_i[n]]
                util += UTIL[true_class][action]

            N = len(y_val)
            abstain_rate = abstain_cnt / N
            answer_rate = answer_cnt / N

            cand = {
                "tau_answer": float(ta),
                "tau_abstainpref": float(ts),
                "utility": float(util),
                "metrics": {
                    "abstain_rate": abstain_rate,
                    "route_answer_rate": answer_rate
                },
            }

            # Track unconstrained best
            if util > best_unconstrained["utility"]:
                best_unconstrained = cand

            # Constraint check
            if (
                abstain_rate <= TARGETS["abstain_rate_max"]
                and answer_rate >= TARGETS["min_answer_rate"]
            ):
                candidates.append(cand)

    # If no candidate meets constraints, fall back to unconstrained best
    if not candidates:
        candidates = [best_unconstrained]

    # Sort by utility desc and keep top-K
    candidates.sort(key=lambda x: x["utility"], reverse=True)
    top = candidates[:TOP_K]

    # Save best and shortlist
    Path(MODEL_DIR, "thresholds.json").write_text(json.dumps(top[0], indent=2), encoding="utf-8")
    Path(MODEL_DIR, "thresholds_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")

    print("Best under constraints (or fallback to unconstrained):")
    print(json.dumps(top[0], indent=2))
    print("\nTop candidates:")
    print(json.dumps(top, indent=2))

if __name__ == "__main__":
    main()
