"""Display-only signal positions, including legacy plots-v1 compatibility.

This file is also shipped verbatim in standalone figure replay bundles.
It never alters samples, PSD bins, metric calculations or source artifacts.
"""

NODES = ('dpd_input', 'pa_input', 'pa_output', 'unknown')
TITLES = {'dpd_input': 'DPD Input', 'pa_input': 'PA Input',
          'pa_output': 'PA Output', 'unknown': 'Unspecified signal position'}


def has_dpd(traces):
    return any(t.get('signal_node') == 'dpd_input' or t.get('stage') == 'u'
               or t.get('role') == 'predistorted' or 'dpd' in t['name'].lower()
               or 'target input' in t['name'].lower() for t in traces)


def signal_node(trace, dpd=False):
    if trace.get('signal_node') in NODES:
        return trace['signal_node']
    role, stage, name = trace.get('role'), trace.get('stage'), trace['name'].lower()
    if stage == 'u' or role == 'predistorted':
        return 'pa_input'
    if stage in ('y', 'reference'):
        return 'pa_output'
    if stage == 'x' or role == 'input':
        return 'dpd_input' if dpd else 'pa_input'
    if role in ('reference', 'primary', 'baseline'):
        return 'pa_output'
    if name.startswith(('u =', 'dpd output', 'pa input')):
        return 'pa_input'
    if name.startswith(('input', 'target input')):
        return 'dpd_input' if dpd else 'pa_input'
    if any(s in name for s in ('pa output', 'with dpd', 'without dpd')):
        return 'pa_output'
    return 'unknown'


def spectrum_groups(traces):
    dpd = has_dpd(traces)
    return [(node, rows) for node in NODES
            if (rows := [t for t in traces if signal_node(t, dpd) == node])]


def spectrum_legend(trace):
    name = trace['name'].lower()
    synthetic = 'synthetic' in trace.get('source', '')
    if name.startswith('ilc ideal'):
        return 'ILC Ideal · waveform-specific'
    if trace.get('role') == 'input':
        return 'Input x'
    if trace.get('role') == 'predistorted':
        return 'Predistorted u'
    if 'linear target' in name:
        return 'Linear target g·x'
    if 'without dpd' in name:
        return 'Without DPD · ' + ('surrogate' if 'surrogate' in name else 'synthetic data' if synthetic else 'measured')
    if 'surrogate' in name or name.startswith('with dpd'):
        return 'With DPD · surrogate'
    if 'with dpd' in name:
        return 'With DPD · measured'
    if 'model output' in name:
        return 'PA model prediction'
    if 'measured pa output' in name:
        return 'Dataset output · synthetic' if synthetic else 'Measured output'
    return trace['name']
