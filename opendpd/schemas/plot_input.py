"""Resource and text limits for stored display data imported from experiments."""
import math


def validate_plot(data):
    if (not isinstance(data, dict) or data.get("version") not in {"plots-v1", "residual-cdf-v1", "declared-power-scan-v1"}
            or data.get("kind") not in {"spectrum", "time", "iq", "am", "amam", "ampm", "error_distribution", "power_scan"}
            or not isinstance(data.get("traces"), list) or len(data["traces"]) > 64):
        raise ValueError("invalid stored plot contract")
    for trace in data["traces"]:
        if not isinstance(trace, dict) or not isinstance(trace.get("name"), str) or len(trace["name"]) > 256:
            raise ValueError("invalid plot trace name")

    def numeric(value):
        if (not isinstance(value, list) or len(value) > 1_000_000
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
            raise ValueError("plot coordinates must be bounded finite numeric arrays")
        return len(value)

    kind = data['kind']
    axis = 'frequency' if kind == 'spectrum' else 'amp_in' if kind in {'am', 'amam', 'ampm'} else 'x'
    if kind in {'time', 'iq'}:
        for trace in data['traces']:
            if numeric(trace.get('i')) != numeric(trace.get('q')):
                raise ValueError("plot coordinate lengths differ")
    else:
        length = numeric(data.get(axis))
        fields = ('psd_db',) if kind == 'spectrum' else ('amp_out', 'phase_deg') if kind in {'am', 'amam', 'ampm'} else ('y',)
        for trace in data['traces']:
            for field in fields:
                if numeric(trace.get(field)) != length:
                    raise ValueError("plot coordinate lengths differ")
    remaining = 2_000_000

    def visit(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 12:
            raise ValueError("stored plot exceeds the structural limit")
        if isinstance(value, str):
            if len(value) > 8192 or any(ord(c) < 32 and c not in "\n\t" for c in value):
                raise ValueError("stored plot text exceeds the text limit")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("stored plot contains non-finite coordinates")
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
    visit(data)
    return data
