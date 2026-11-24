
import json
import random
from pathlib import Path
from collections import defaultdict

SOURCE = Path("data/processed/candidates.jsonl")
OUTPUT = Path("data/processed")

SEED = 42


def main():
    random.seed(SEED)

    groups = defaultdict(list)

    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row["metadata"]["review_status"] != "approved":
                continue

            source_id = row["metadata"]["source_id"]
            groups[source_id].append(row)

    document_ids = list(groups.keys())
    random.shuffle(document_ids)

    n = len(document_ids)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": document_ids[:train_end],
        "validation": document_ids[train_end:val_end],
        "test": document_ids[val_end:]
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, ids in splits.items():
        path = OUTPUT / f"{name}.jsonl"
        count = 0

        with path.open("w", encoding="utf-8") as out:
            for source_id in ids:
                for row in groups[source_id]:
                    row["metadata"]["split"] = name

                    out.write(
                        json.dumps(row, ensure_ascii=False)
                        + "\n"
                    )

                    count += 1

        print(
            f"{name}: {count} examples "
            f"from {len(ids)} documents"
        )


if __name__ == "__main__":
    main()
