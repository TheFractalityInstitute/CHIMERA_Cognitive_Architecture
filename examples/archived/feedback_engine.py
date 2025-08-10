"""
Feedback Loop Engine for CHIMERA Crystallization System
Fetches relevant insights from archive and injects them into reasoning prompts.
"""

import os
import json

ARCHIVE_FOLDER = "archive"
INDEX_FILE = os.path.join(ARCHIVE_FOLDER, "index.json")

def load_index():
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def retrieve_matching_insights(keyword=None, tier=None):
    insights = []
    index = load_index()
    for uid, meta in index.items():
        match = True
        if keyword and keyword.lower() not in meta["title"].lower():
            match = False
        if tier and tier != meta.get("tier", ""):
            match = False
        if match:
            filepath = os.path.join(ARCHIVE_FOLDER, meta["file"])
            with open(filepath, "r") as f:
                insights.append(json.load(f))
    return insights

def inject_into_prompt(prompt, keyword=None, tier=None):
    insights = retrieve_matching_insights(keyword=keyword, tier=tier)
    if not insights:
        return prompt
    prompt += "\n\n# Retrieved Crystallized Insights:\n"
    for insight in insights:
        prompt += f"- ({insight['tier']}) {insight['title']}: {insight['essence']}\n"
    return prompt

# Example usage
if __name__ == "__main__":
    base_prompt = "What metaphysical structures govern selfhood in distributed agents?"
    enhanced = inject_into_prompt(base_prompt, keyword="selfhood", tier="T7")
    print(enhanced)
