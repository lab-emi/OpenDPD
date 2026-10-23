"""Readable name of a test-model signal, for pickers and dataset captures (the generated ``preset_id`` is an identifier, not a name)."""
from __future__ import annotations


def recipe_label(recipe) -> str:
    """``NR-FR1-TM3.1a · 100 MHz · 30 kHz · FDD``; several carriers read ``2 × 100 MHz`` (equal) or ``20+100 MHz`` (listed in
    ascending frequency order), mixed numerologies ``15/30 kHz``, rank 2 appends ``2 layers``. At most 16 carriers: under 100 characters."""
    widths = [carrier.bandwidth_mhz for carrier in recipe.carriers]
    if len(widths) == 1:
        bandwidth = f"{widths[0]} MHz"
    elif len(set(widths)) == 1:
        bandwidth = f"{len(widths)} × {widths[0]} MHz"
    else:
        bandwidth = "+".join(str(width) for width in widths) + " MHz"
    spacing = "/".join(str(scs) for scs in sorted({carrier.scs_khz for carrier in recipe.carriers}))
    parts = [recipe.model, bandwidth, f"{spacing} kHz", recipe.duplex.upper()]
    if recipe.layers == 2:
        parts.append("2 layers")
    return " · ".join(parts)
