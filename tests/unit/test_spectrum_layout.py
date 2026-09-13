import numpy as np

from opendpd.core.plots import spectrum
from opendpd.core.spectrum_layout import spectrum_groups, spectrum_legend
from opendpd.services.figure_render import split_spectrum_panels


def test_dpd_positions_preserve_the_output_comparisons_and_the_actual_bins():
    names = ['target input x', 'u = DPD(x)', 'linear target gain*x',
             'with DPD: PA_surrogate(u)', 'surrogate without DPD', 'measured PA without DPD']
    roles = dict(zip(names, ['input', 'predistorted', 'reference', 'primary', 'baseline', 'baseline']))
    z = np.exp(2j * np.pi * np.arange(128) / 16)
    data = spectrum({name: np.column_stack((z.real, z.imag)) * (i + 1) for i, name in enumerate(names)}, roles,
                    sample_rate_hz=1e6, nperseg=128, bandwidth_hz=1e5, input_node='dpd_input')
    groups = spectrum_groups(data['traces'])
    assert [(node, [t['name'] for t in traces]) for node, traces in groups] == [
        ('dpd_input', names[:1]), ('pa_input', names[1:2]), ('pa_output', names[2:])]
    # New metadata and legacy inference must resolve to the same locations.
    legacy = [{k: v for k, v in t.items() if k != 'signal_node'} for t in data['traces']]
    assert [(node, [t['psd_db'] for t in traces]) for node, traces in spectrum_groups(legacy)] == [
        (node, [t['psd_db'] for t in traces]) for node, traces in groups]
    panels = split_spectrum_panels([dict(kind='spectrum', traces=[dict(run_id='r', trace_name=n, visible=True) for n in names], x_range=[-1, 1])], {'r/spectrum': data})
    assert [p['signal_node'] for p in panels] == ['dpd_input', 'pa_input', 'pa_output']
    assert all(p['x_range'] == [-1, 1] for p in panels)
    assert len(panels[2]['traces']) == 4


def test_pa_only_and_synthetic_legacy_labels_are_not_claimed_as_dpd_or_measured():
    traces = [dict(name='PA input x', role='input', stage='x'),
              dict(name='measured PA output', role='reference', stage='reference', source='synthetic dataset'),
              dict(name='PA model output', role='primary', stage='y')]
    assert [node for node, _ in spectrum_groups(traces)] == ['pa_input', 'pa_output']
    assert spectrum_legend(traces[1]) == 'Dataset output · synthetic'
    assert spectrum_groups([dict(name='Unlabelled probe')])[0][0] == 'unknown'
