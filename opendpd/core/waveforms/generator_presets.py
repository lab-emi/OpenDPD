"""Uncoded engineering stimuli with traceable NR / WLAN numerology tables."""
from functools import lru_cache
import math

from opendpd.schemas.signal_generator import GeneratorConfig, GeneratorPreset

# TS 38.104 V18.9.0 tables 5.3.2-1 / 5.3.2-2. See docs/guides/signal-presets.md.
_NR_BANDWIDTHS = (3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
_NR_RBS = {
    15: (15, 25, 52, 79, 106, 133, 160, 188, 216, 242, 270, 0, 0, 0, 0, 0),
    30: (0, 11, 24, 38, 51, 65, 78, 92, 106, 119, 133, 162, 189, 217, 245, 273),
    60: (0, 0, 11, 18, 24, 31, 38, 44, 51, 58, 65, 79, 93, 107, 121, 135),
}
_WLAN_TONES = {20: (242, 106, 52, 26), 40: (484, 242, 106, 52),
               80: (996, 484, 242, 106), 160: (1992, 996, 484, 242),
               320: (3984, 1992, 996, 484)}


@lru_cache(maxsize=1)
def _catalog() -> tuple[GeneratorPreset, ...]:
    items = []

    def add(identifier, family, label, description, numerology="Custom", **values):
        rate = values.get("sample_rate_hz", 80e6)
        values.setdefault("n_samples", max(16384, min(196608, round(rate * .00025))))
        config = GeneratorConfig(preset_id=identifier, **values)
        items.append(GeneratorPreset(preset_id=identifier, family=family, label=label,
            description=description, numerology=numerology,
            channel_count=len(config.channel_subcarriers), config=config))

    nr_tables = [("FR1", scs, tuple(zip(_NR_BANDWIDTHS, rbs))) for scs, rbs in _NR_RBS.items()]
    nr_tables += [("FR2-1", 60, ((50, 66), (100, 132), (200, 264))),
                  ("FR2-1", 120, ((50, 32), (100, 66), (200, 132), (400, 264)))]
    for region, scs, bands in nr_tables:
        for bandwidth, rbs in bands:
            if not rbs:
                continue
            fft = 2**math.ceil(math.log2(bandwidth * 1000 / scs))
            for count in (1, 2, 4, 8):
                for order in (4, 16, 64, 256, 1024):
                    identifier = f"nr-{region.lower()}-{scs}-{bandwidth}-q{order}-c{count}"
                    if region == "FR1" and scs == 30 and count == 1 and order == 64 and bandwidth in (20, 100):
                        identifier = f"nr-{bandwidth}"
                    if region == "FR2-1" and scs == 120 and bandwidth == 100 and count == 1 and order == 64:
                        identifier = "nr-fr2"
                    add(identifier, "nr", f"{region} · {bandwidth} MHz · {order}-QAM · {count} ch",
                        f"{scs} kHz SCS · normal CP · {rbs // count} RB per channel · uncoded payload",
                        numerology=f"{region} · {scs} kHz", sample_rate_hz=fft*scs*4000,
                        bandwidth_hz=bandwidth*1e6, carrier_frequency_hz=3.5e9 if region == "FR1" else 28e9,
                        fft_size=fft, channel_subcarriers=[rbs//count*12]*count,
                        channel_modulations=[order]*count, channel_power_db=[0.]*count,
                        cp_mode="nr_normal", pilot_spacing=12)
    for family, bandwidths, orders in [("wifi6", (20, 40, 80, 160), (2, 4, 16, 64, 256, 1024)),
                                     ("wifi7", (20, 40, 80, 160, 320), (2, 4, 16, 64, 256, 1024, 4096))]:
        for bandwidth in bandwidths:
            for count, tones in zip((1, 2, 4, 8), _WLAN_TONES[bandwidth]):
                for order in orders:
                    identifier = f"{family}-{bandwidth}-q{order}-c{count}"
                    if count == 1 and order == orders[-1]:
                        identifier = f"{family}-{bandwidth}"
                    # RU-sized allocations use generic placement/pilots, not packet RU bitmaps.
                    add(identifier, family, f"{bandwidth} MHz · {order}-QAM · {count} ch",
                        f"78.125 kHz SCS · 0.8 µs guard · {tones} tones per channel · uncoded payload",
                        numerology="HE · 78.125 kHz" if family == "wifi6" else "EHT · 78.125 kHz",
                        sample_rate_hz=bandwidth*4e6, bandwidth_hz=bandwidth*1e6,
                        carrier_frequency_hz=5.8e9 if family == "wifi6" else 6.1e9,
                        fft_size=bandwidth*256//20, cp_samples=bandwidth*16//20,
                        channel_subcarriers=[tones]*count, channel_modulations=[order]*count,
                        channel_power_db=[0.]*count, channel_gap_bins=4 if count > 1 else 0,
                        pilot_spacing=16)
    add("custom-ofdm", "custom", "Custom OFDM / OFDMA", "Independent channel allocations, pilots and modulation")
    add("custom-qam", "custom", "Single-carrier QAM / PSK", "Root-raised-cosine pulse shaping",
        waveform="qam", sample_rate_hz=80e6, bandwidth_hz=20e6)
    add("custom-tone", "custom", "Single tone", "Complex sinusoid for gain and phase checks", waveform="tone")
    add("custom-multitone", "custom", "Multitone", "Equally spaced tones with seeded random phases", waveform="multitone")
    add("custom-chirp", "custom", "Linear chirp", "Complex baseband frequency sweep", waveform="chirp")
    add("custom-psk", "custom", "8-PSK", "Gray-labeled phase modulation with RRC pulse shaping", waveform="psk")
    add("custom-fsk", "custom", "Binary FSK", "Continuous-phase frequency modulation", waveform="fsk", samples_per_symbol=16)
    add("custom-gfsk", "custom", "Gaussian FSK", "Continuous phase · configurable Gaussian BT", waveform="gfsk", samples_per_symbol=16)
    add("custom-noise", "custom", "Band-limited noise", "Complex Gaussian noise for spectral loading", waveform="noise")
    add("custom-dft-ofdm", "custom", "DFT-spread OFDM", "Single-carrier-like envelope · uncoded uplink stimulus",
        dft_spreading=True, pilot_mode="none")
    # A familiar starting point is first; the UI sorts matrix axes numerically.
    items.sort(key=lambda p: p.preset_id != "nr-20")
    return tuple(items)


def presets() -> list[GeneratorPreset]:
    return list(_catalog())


def coverage(config: GeneratorConfig) -> str:
    preset = next((p for p in presets() if p.preset_id == config.preset_id), None)
    if preset is None or preset.family == "custom":
        return "custom"
    keys = ("waveform", "sample_rate_hz", "bandwidth_hz", "fft_size", "oversampling",
            "channel_subcarriers", "channel_gap_bins", "cp_mode", "cp_samples", "dc_null", "dft_spreading")
    if any(getattr(config, k) != getattr(preset.config, k) for k in keys):
        return "custom"
    return "numerology"
