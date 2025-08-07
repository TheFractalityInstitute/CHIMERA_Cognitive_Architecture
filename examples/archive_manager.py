"""
Archive Manager for CHIMERA Crystallization Engine
Handles persistence, indexing, and retrieval of crystallized insights.
"""

import os
import json
from datetime import datetime

ARCHIVE_FOLDER = "archive"
INDEX_FILE = os.path.join(ARCHIVE_FOLDER, "index.json")

def ensure_archive():
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w") as f:
            json.dump({}, f)

def load_index():
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

def add_crystallized_insight(insight, filename=None):
    ensure_archive()
    index = load_index()

    uid = insight["id"]
    if not filename:
        filename = f"{uid}.json"
    filepath = os.path.join(ARCHIVE_FOLDER, filename)

    with open(filepath, "w") as f:
        json.dump(insight, f, indent=2)

    index[uid] = {
        "file": filename,
        "title": insight.get("title", ""),
        "tier": insight.get("tier", "T0"),
        "timestamp": insight.get("timestamp", datetime.utcnow().isoformat())
    }
    save_index(index)
    return filepath

def retrieve_by_id(uid):
    ensure_archive()
    index = load_index()
    if uid not in index:
        return None
    filepath = os.path.join(ARCHIVE_FOLDER, index[uid]["file"])
    with open(filepath, "r") as f:
        return json.load(f)

# Example usage
if __name__ == "__main__":
    example = {
        "id": "crystal_001",
        "title": "Fractal Reciprocity Principle",
        "essence": "What is given in one form echoes through all tiers.",
        "tier": "T5",
        "source": "Fractality Core Doctrine",
        "timestamp": datetime.utcnow().isoformat()
    }
    add_crystallized_insight(example)
    loaded = retrieve_by_id("crystal_001")
    print("Retrieved:", loaded["essence"])
