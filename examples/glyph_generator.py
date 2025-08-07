"""
CHIMERA Glyph Generator
Generates symbolic glyph representations from crystallized metadata.
"""

def generate_glyph(tier, shape="triadic", flow="↔", center="⊙"):
    # Tier glyph is number of ● for echo depth
    echo = "●" * int(tier[1]) if tier.startswith("T") else ""

    # Shape symbol
    shape_map = {
        "triadic": "△",
        "structural": "▭",
        "subjective": "◯",
        "spiral": "⟳"
    }
    shape_sym = shape_map.get(shape, "*")

    # Compose final glyph
    glyph = f"{center}{shape_sym}{flow}{echo}"
    return glyph

def create_glyph_metadata(title, tier, shape, flow, source):
    symbol = generate_glyph(tier, shape, flow)
    return {
        "glyph_id": f"glyph-{title.lower().replace(' ', '-')}",
        "symbol": symbol,
        "tier": tier,
        "source": source,
        "meaning": f"{title} represented as {symbol}"
    }

# Example usage
if __name__ == "__main__":
    glyph = create_glyph_metadata("Triadic Ontology Principle", "T3", "triadic", "↔", "Fractality Core")
    print(glyph)
