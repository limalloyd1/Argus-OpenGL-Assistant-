
import json
import requests
from pathlib import Path

INPUT = Path("data/processed/chunks.jsonl")
OUTPUT = Path("data/processed/candidates.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"
TEACHER_MODEL = "qwen2.5:3b"


SYSTEM_PROMPT = """
You are an expert OpenGL and computer graphics educator.

Your task is to generate supervised fine-tuning examples
for an OpenGL programming assistant.

You will receive a documentation passage.

Generate up to 4 high-quality question and answer pairs.

Available task types:
- conceptual_explanation
- code_analysis
- implementation
- troubleshooting
- comparative_analysis

Rules:
1. Use only information supported by the supplied passage.
2. Do not invent APIs, functions, or technical facts.
3. Preserve C++, GLSL, and OpenGL terminology.
4. Make questions natural and technically meaningful.
5. Answers should explain the underlying concepts.
6. If code is supplied, you may ask about that code.
7. Do not force a task category when the passage
   does not support it.
8. Avoid duplicate questions.

Return valid JSON in this format:

{
  "examples": [
    {
      "task_type": "conceptual_explanation",
      "question": "Question here",
      "answer": "Answer here"
    }
  ]
}
"""


def generate_examples(chunk):
    prompt = f"""
DOCUMENT TITLE: {chunk['title']}
SECTION: {chunk['section']}
TOPIC: {chunk['topic']}

DOCUMENTATION:
{chunk['content']}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": TEACHER_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "options": {
                "temperature": 0.2
            }
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    parsed = json.loads(result["message"]["content"])

    return parsed.get("examples", [])


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with INPUT.open(encoding="utf-8") as source, \
         OUTPUT.open("w", encoding="utf-8") as target:

        for line in source:
            chunk = json.loads(line)

            try:
                examples = generate_examples(chunk)

                for example in examples:
                    question = example.get("question", "").strip()
                    answer = example.get("answer", "").strip()
                    task = example.get("task_type", "").strip()

                    if not question or not answer:
                        continue

                    record = {
                        "messages": [
                            {
                                "role": "user",
                                "content": question
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ],
                        "metadata": {
                            "source_id": chunk["source_id"],
                            "chunk_id": chunk["id"],
                            "source": chunk["source"],
                            "section": chunk["section"],
                            "topic": chunk["topic"],
                            "task_type": task,
                            "source_url": chunk["url"],
                            "review_status": "pending"
                        }
                    }

                    target.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )

                    total += 1

                print(f"Processed: {chunk['id']}")

            except Exception as error:
                print(f"Failed {chunk['id']}: {error}")

    print(f"\nGenerated {total} candidate examples.")


if __name__ == "__main__":
    main()
