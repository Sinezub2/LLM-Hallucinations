# src/make_labels.py
import csv, json, random
from pathlib import Path
from .label_map import CATEGORY_TO_LABEL

IN_PATH = Path("data/prompts.csv")      # columns: category,instruction[,context]
OUT_PATH = Path("data/training.jsonl")

# Controls
RANDOM_SEED = 13
MAX_ABSTAIN_RATIO = 0.25       # cap ABSTAIN examples at 25% of dataset
SYNTH_ABSTAIN_PROB = 0.30      # chance to create a truncated (underspecified) ABSTAIN

def synthesize_abstain_rows(cat: str, instr: str, ctx: str | None):
    """Generate at most one ABSTAIN example per row, and only when justified."""
    # 1) If task relies on context but it's missing → ABSTAIN
    if cat in ("summarization", "information_extraction") and not ctx:
        return [{"text": instr, "label": "ABSTAIN"}]

    # 2) Otherwise, *occasionally* truncate long instructions to simulate underspecification
    if len(instr) > 80 and random.random() < SYNTH_ABSTAIN_PROB:
        truncated = instr[: max(40, len(instr)//2)].rstrip() + " ..."
        return [{"text": truncated, "label": "ABSTAIN"}]

    return []

def map_row(cat: str, instr: str, ctx: str | None):
    """Primary label comes strictly from category (no trigger words)."""
    if cat == "summarization" and not ctx:
        return "ABSTAIN"  # explicit underspecified summarization
    return CATEGORY_TO_LABEL.get(cat, "ABSTAIN_PREFER")

def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH}"
    random.seed(RANDOM_SEED)

    rows_out = []
    count_in = 0
    with IN_PATH.open("r", encoding="utf-8") as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            count_in += 1
            cat = (row.get("category") or "").strip()
            instr = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip() or None
            if not instr:
                continue

            label = map_row(cat, instr, ctx)
            rows_out.append({"text": instr, "label": label})

            # add at most one synthesized ABSTAIN candidate
            rows_out.extend(synthesize_abstain_rows(cat, instr, ctx))

    # Enforce ABSTAIN cap to avoid over-abstaining policy
    total = len(rows_out)
    max_abstain = int(total * MAX_ABSTAIN_RATIO)
    abstain_rows = [r for r in rows_out if r["label"] == "ABSTAIN"]
    non_abstain   = [r for r in rows_out if r["label"] != "ABSTAIN"]

    if len(abstain_rows) > max_abstain:
        random.shuffle(abstain_rows)
        abstain_rows = abstain_rows[:max_abstain]

    final_rows = non_abstain + abstain_rows
    random.shuffle(final_rows)

    # write
    with OUT_PATH.open("w", encoding="utf-8") as fw:
        for r in final_rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report class balance
    from collections import Counter
    cnt = Counter([r["label"] for r in final_rows])
    print(f"Inputs read: {count_in}")
    print(f"Outputs written: {len(final_rows)} (cap ABSTAIN ≤ {MAX_ABSTAIN_RATIO*100:.0f}%)")
    print("Label counts:", dict(cnt))

if __name__ == "__main__":
    main()
