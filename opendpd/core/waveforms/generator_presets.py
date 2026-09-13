"""Engineering stimuli with documented NR / WLAN numerologies, not encoded packets."""
from opendpd.schemas.signal_generator import GeneratorConfig, GeneratorPreset


def presets() -> list[GeneratorPreset]:
    items = []

    def add(identifier, family, label, description, **values):
        items.append(GeneratorPreset(preset_id=identifier, family=family, label=label,
            description=description, config=GeneratorConfig(preset_id=identifier, **values)))

    for bandwidth, rate, fft, carriers in [(20, 30.72, 1024, 612), (100, 122.88, 4096, 3276)]:
        add(f"nr-{bandwidth}", "nr", f"FR1 · {bandwidth} MHz", "30 kHz SCS · normal CP · 64-QAM",
            sample_rate_hz=rate*4e6, bandwidth_hz=bandwidth*1e6, fft_size=fft,
            channel_subcarriers=[carriers], cp_mode="nr_normal", pilot_spacing=12)
    add("nr-fr2", "nr", "FR2 · 100 MHz", "120 kHz SCS · normal CP · 64-QAM",
        sample_rate_hz=491.52e6, bandwidth_hz=100e6, carrier_frequency_hz=28e9,
        fft_size=1024, channel_subcarriers=[792], cp_mode="nr_normal", pilot_spacing=12)
    for family, order, bandwidths in [("wifi6", 1024, (20, 40, 80, 160)), ("wifi7", 4096, (20, 40, 80, 160, 320)), ("wifi8", 4096, (80, 160, 320))]:
        for bandwidth in bandwidths:
            # Continuous payload numerology. Multiple 996-tone allocations at 160/320 MHz
            # have generic gaps and pilots; no standard RU bitmap or signaling is claimed.
            blocks = max(1, bandwidth // 80)
            carriers = {20: 242, 40: 484}.get(bandwidth, 996)
            add(f"{family}-{bandwidth}", family, f"{bandwidth} MHz · {order}-QAM",
                "78.125 kHz SCS · 0.8 µs guard interval · continuous OFDM payload",
                sample_rate_hz=bandwidth*4e6, bandwidth_hz=bandwidth*1e6,
                carrier_frequency_hz=5.8e9 if family == "wifi6" else 6.1e9,
                fft_size=bandwidth*256//20, cp_samples=bandwidth*16//20,
                channel_subcarriers=[carriers]*blocks, channel_modulations=[order]*blocks,
                channel_power_db=[0.]*blocks, channel_gap_bins=4 if blocks > 1 else 0,
                pilot_spacing=16)
    add("custom-ofdm", "custom", "Custom OFDM / OFDMA", "Independent channel allocations, pilots and modulation")
    add("custom-qam", "custom", "Single-carrier QAM / PSK", "Root-raised-cosine pulse shaping",
        waveform="qam", sample_rate_hz=80e6, bandwidth_hz=20e6)
    add("custom-tone", "custom", "Single tone", "Complex sinusoid for gain and phase checks", waveform="tone")
    add("custom-multitone", "custom", "Multitone", "Equally spaced tones with seeded random phases", waveform="multitone")
    add("custom-chirp", "custom", "Linear chirp", "Complex baseband frequency sweep", waveform="chirp")
    return items


def coverage(config: GeneratorConfig) -> str:
    preset = next((p for p in presets() if p.preset_id == config.preset_id), None)
    if preset is None or preset.family == "custom":
        return "custom"
    keys = ("waveform", "sample_rate_hz", "bandwidth_hz", "fft_size", "oversampling",
            "channel_subcarriers", "channel_gap_bins", "cp_mode", "cp_samples", "dc_null")
    if any(getattr(config, k) != getattr(preset.config, k) for k in keys):
        return "custom"
    return "experimental" if preset.family == "wifi8" else "numerology"
