# solution.py
import json
from pathlib import Path
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer

# Paths
MODEL_DIR = Path("models")
IN_PATH = Path("input.json")
OUT_PATH = Path("output.json")

# Router load (once)
EMBED_MODEL = "intfloat/e5-small-v2"
from pathlib import Path
from llama_cpp import Llama

LLM_PATH = Path("models/qwen2.5-0.5b-instruct-q5_k_m.gguf")
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(LLM_PATH),
            n_ctx=2048,      # reduce to 1024 if memory tight
            n_threads=4,     # set to your CPU cores
            seed=42,
            verbose=False
        )
    return _llm

SYSTEM_RU = (
    "Ты помощник, отвечающий кратко, корректно и без выдумок. "
    "Пиши на русском. Если данных не хватает — уточняй."
)

def slm_answer(prompt_ru: str) -> str:
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_RU},
        {"role": "user", "content": prompt_ru.strip()},
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2, top_p=0.9, max_tokens=256, stop=[]
    )
    return out["choices"][0]["message"]["content"].strip()

def load_router():
    meta = load(MODEL_DIR / "meta.joblib")
    clf = load(MODEL_DIR / "clf.joblib")
    calib = load(MODEL_DIR / "calib.joblib")
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    embedder = SentenceTransformer(EMBED_MODEL)
    le = meta["label_encoder"]
    class_names = list(le.classes_)
    idx = {name: i for i, name in enumerate(class_names)}
    return embedder, clf, calib, thresholds, idx

def calibrate_probs(raw, cal_models):
    return np.stack([cal_models[c].transform(raw[:, c]) for c in range(raw.shape[1])], axis=1)

def decide_single(text, embedder, clf, calib, thresholds, idx):
    X = embedder.encode([text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    raw = clf.predict_proba(X)
    prob = calibrate_probs(raw, calib["cal_models"])

    pA = prob[0, idx.get("ANSWER_PREFER", -1)] if "ANSWER_PREFER" in idx else 0.0
    pS = prob[0, idx.get("ABSTAIN_PREFER", -1)] if "ABSTAIN_PREFER" in idx else 0.0

    if pA >= thresholds["tau_answer"]:
        return "ANSWER_PREFER", float(pA), float(pS)
    elif pS >= thresholds["tau_abstainpref"]:
        return "ABSTAIN_PREFER", float(pA), float(pS)
    else:
        return "ABSTAIN", float(pA), float(pS)

def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH.resolve()}"
    prompts = json.loads(IN_PATH.read_text(encoding="utf-8"))
    assert isinstance(prompts, list), "input.json must be a JSON list of strings"

    embedder, clf, calib, thresholds, idx = load_router()

    outputs = []
    for q in prompts:
        q_str = q if isinstance(q, str) else json.dumps(q, ensure_ascii=False)
        decision, pA, pS = decide_single(q_str, embedder, clf, calib, thresholds, idx)

    if decision == "ANSWER_PREFER":
        outputs.append(slm_answer(q_str))  # <— real Russian answer now
    else:
        outputs.append("Воздерживаюсь: недостаточно уверенности без дополнительных источников.")


    OUT_PATH.write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
