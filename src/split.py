import json, random
from pathlib import Path

IN_PATH = Path("data/training.jsonl")
OUT_DIR = Path("data")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH}"
    rows = [json.loads(l) for l in IN_PATH.read_text(encoding="utf-8").splitlines()]
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    parts = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train+n_val],
        "test": rows[n_train+n_val:],
    }
    for name, items in parts.items():
        Path(OUT_DIR, f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items),
            encoding="utf-8"
        )
        print(name, len(items))

if __name__ == "__main__":
    main()
