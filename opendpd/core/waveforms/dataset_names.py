"""Compact, deterministic spec summaries; names never define waveform identity."""
def _number(value):
    if 0 < abs(value) < .001:
        mantissa, exponent = f"{value:.2e}".split("e")
        return mantissa.rstrip("0").rstrip(".").replace(".", "p") + "e" + str(int(exponent))
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _range(values):
    low, high = min(values), max(values)
    return _number(low) if low == high else f"{_number(low)}-{_number(high)}"


def standard_model_family(model):
    """Family token of an NR standard test model: "nr" + the model id without "NR-FR1-"/"NR-", lower case, "." as "p", no "-".

    NR-FR1-TM3.1a -> nrtm3p1a, NR-FR1-TM1.1 -> nrtm1p1, NR-N-TM -> nrntm. frontend/src/utils/datasetNames.ts (testModelFamily)
    mirrors this rule; both test suites assert the same strings.
    """
    for prefix in ("NR-FR1-", "NR-"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    return "nr" + model.lower().replace(".", "p").replace("-", "")


def _family(config):
    if config.waveform != "ofdm":
        return config.waveform
    recipe = getattr(config, "nr_test_model", None)
    if recipe is not None:
        return standard_model_family(recipe.model)
    return ("nr" if config.preset_id.startswith("nr-") else
            "w6" if config.preset_id.startswith("wifi6-") else
            "w7" if config.preset_id.startswith("wifi7-") else config.waveform)


def dataset_name(configs, role="in", model_id=None):
    families = sorted({_family(c) for c in configs})
    parts = ["-".join(families) if len(families) <= 2 else "mix", "bw" + _range([c.bandwidth_hz / 1e6 for c in configs]) + "M"]
    qam = [v for c in configs for v in (c.channel_modulations if c.waveform == "ofdm" else [c.modulation_order] if c.waveform == "qam" else [])]
    psk = [c.psk_order for c in configs if c.waveform == "psk"]
    ofdm = [c for c in configs if c.waveform == "ofdm"]
    if qam:
        parts.append("q" + _range(qam))
    if psk:
        parts.append("p" + _range(psk))
    if ofdm:
        parts.append("c" + _range([len(c.channel_subcarriers) for c in ofdm]))
        spacing = [c.sample_rate_hz / (c.fft_size * c.oversampling * 1000) for c in ofdm]
        if len(set(spacing)) > 1:
            parts.append("s" + _range(spacing) + "k")
    parts.append("n" + str(len(configs)))
    if model_id:
        parts.append(model_id[:20])
    return f"syn_pa_{role}_" + "_".join(parts)
