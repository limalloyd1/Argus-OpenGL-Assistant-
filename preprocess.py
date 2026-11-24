
import json
import re
from pathlib import Path

RAW = Path("data/raw/pages.jsonl")
OUTPUT = Path("data/processed/chunks.jsonl")

MAX_CHARS = 4500
MIN_CHARS = 100


def load_documents():
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_sections(markdown):
    sections = []
    heading = "Introduction"
    content = []

    for line in markdown.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)", line)

        if match:
            if content:
                sections.append((heading, "\n".join(content).strip()))

            heading = match.group(1).strip()
            content = []
        else:
            content.append(line)

    if content:
        sections.append((heading, "\n".join(content).strip()))

    return sections


def pack_paragraphs(text):
    # Protect fenced code blocks from paragraph splitting.
    blocks = []

    def protect(match):
        blocks.append(match.group(0))
        return f"CODEBLOCKPLACEHOLDER{len(blocks)-1}"

    protected = re.sub(
        r"```[\s\S]*?```",
        protect,
        text
    )

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", protected)
        if p.strip()
    ]

    result = []
    current = []

    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])

        if len(candidate) <= MAX_CHARS:
            current.append(paragraph)
        else:
            if current:
                result.append("\n\n".join(current))

            current = [paragraph]

    if current:
        result.append("\n\n".join(current))

    # Restore code blocks.
    restored = []

    for chunk in result:
        for i, block in enumerate(blocks):
            chunk = chunk.replace(
                f"CODEBLOCKPLACEHOLDER{i}",
                block
            )
        restored.append(chunk)

    return restored


def preprocess():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with OUTPUT.open("w", encoding="utf-8") as out:
        for doc in load_documents():
            chunk_number = 0

            for heading, section_text in extract_sections(
                doc["markdown"]
            ):
                if not section_text:
                    continue

                for chunk in pack_paragraphs(section_text):
                    if len(chunk.strip()) < MIN_CHARS:
                        continue

                    record = {
                        "id": f"{doc['id']}_{chunk_number:04d}",
                        "source_id": doc["id"],
                        "source": "LearnOpenGL",
                        "url": doc["url"],
                        "title": doc["title"],
                        "section": doc["section"],
                        "topic": heading,
                        "content": chunk,
                        "code_refs": doc.get("code_refs", []),
                        "n_code_blocks": doc.get("n_code_blocks", 0)
                    }

                    out.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )

                    chunk_number += 1
                    total += 1

    print(f"Generated {total} chunks.")


if __name__ == "__main__":
    preprocess()
