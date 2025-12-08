import json
from pathlib import Path
from typing import Dict


def ensure_json_file(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("{}", encoding="utf-8")


def save_topic_text(json_path: str, topic: str, text: str):
    ensure_json_file(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data[topic] = text
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_texts_by_topics(json_path: str) -> Dict[str, str]:
    ensure_json_file(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)