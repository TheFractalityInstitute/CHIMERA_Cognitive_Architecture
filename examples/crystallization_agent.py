"""
Tier 1 Crystallization Agent Prototype
Monitors text input for high-resonance patterns and crystallizes them.
"""

import json
import re
from datetime import datetime

def detect_resonant_insight(text):
    if any(keyword in text.lower() for keyword in ["core", "truth", "triadic", "ontology", "resonance"]):
        return True
    return False

def crystallize(text, source="CHIMERA Input Stream"):
    return {
        "id": f"insight-{datetime.utcnow().isoformat()}",
        "timestamp": datetime.utcnow().isoformat(),
        "essence": text.strip(),
        "source": source,
        "tags": ["auto", "crystallized"],
        "tier": "unspecified"
    }

def run_agent(log_stream):
    crystals = []
    for line in log_stream:
        if detect_resonant_insight(line):
            crystal = crystallize(line)
            crystals.append(crystal)
    return crystals

# Example usage
if __name__ == "__main__":
    logs = [
        "This is a basic thought.",
        "The triadic ontology reveals deep structure.",
        "Fractal resonance aligns subjective with objective modes."
    ]
    results = run_agent(logs)
    with open("crystallized_output.json", "w") as f:
        json.dump(results, f, indent=4)
