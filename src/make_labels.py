import csv, json, random
from pathlib import Path
from .label_map import CATEGORY_TO_LABEL, ABSTAIN_TRIGGERS


IN_PATH = Path("data/prompts.csv")      # columns: category,instruction[,context]
OUT_PATH = Path("data/training.jsonl")

def has_trigger(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ABSTAIN_TRIGGERS)

def synthesize_abstain_rows(cat: str, instr: str, ctx: str | None):
    rows = []
    # If task relies on context but it's missing → abstain sample
    if cat in ("summarization", "information_extraction") and not ctx:
        rows.append({"text": instr, "label": "ABSTAIN"})
    # Truncate to make it underspecified
    if len(instr) > 60:
        truncated = instr[: max(20, len(instr)//2)] + "..."
        rows.append({"text": truncated, "label": "ABSTAIN"})
    return rows

def map_row(cat: str, instr: str, ctx: str | None):
    label = CATEGORY_TO_LABEL.get(cat, "ABSTAIN_PREFER")
    if cat == "summarization" and not ctx:
        return "ABSTAIN"
    if has_trigger(instr):
        label = "ABSTAIN_PREFER"
    return label

def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH}"
    random.seed(13)
    count_in, count_out = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as fw, IN_PATH.open("r", encoding="utf-8") as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            count_in += 1
            cat = (row.get("category") or "").strip()
            instr = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip() or None
            if not instr:
                continue
            label = map_row(cat, instr, ctx)
            fw.write(json.dumps({"text": instr, "label": label}, ensure_ascii=False) + "\n")
            count_out += 1
            for extra in synthesize_abstain_rows(cat, instr, ctx):
                fw.write(json.dumps(extra, ensure_ascii=False) + "\n")
                count_out += 1
    print(f"Wrote {count_out} rows from {count_in} inputs to {OUT_PATH}")

if __name__ == "__main__":
    main()
