"""Exported as reproduce.py. Verify first, then rebuild metrics and plot coordinates."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def reproduce(root, destination, use_bundled_source=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a new, nonexistent workspace directory; existing work is never replaced.')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundle hash mismatch: {name}')
    if use_bundled_source:
        import sys
        sys.path.insert(0, str(root / 'runtime-source'))
    from opendpd.services import evaluation, experiments, measurements, packages, figure_render
    from opendpd.services.workspace import Workspace, PACKAGE_ROOT, read_json, software_provenance
    from opendpd.schemas import TaskType
    for name, expected in json.loads((root / 'implementation.json').read_text()).items():
        path = (PACKAGE_ROOT / name).resolve()
        if not path.is_relative_to(PACKAGE_ROOT.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'OpenDPD source differs: {name}. Install the exact recorded implementation before re-evaluation.')
    import numpy as np
    figure = read_json(root / 'figure.json')
    original_plots = read_json(root / 'plot-data.json')
    rebuilt = copy.deepcopy(original_plots)
    report = {'protocol': 'metric-and-figure-reproduction-v1', 'passed': True, 'retrained': False,
              'implementation': 'bundled' if use_bundled_source else 'installed',
              'relative_tolerance': 1e-4, 'absolute_tolerance': 1e-5,
              'tolerance_status': 'proposed CPU checkpoint tolerance, pending maintainer approval; also applied to display coordinates',
              'environment': software_provenance().model_dump(mode='json'), 'runs': []}
    destination.mkdir(parents=True)
    for binding in figure['bindings']:
        run = binding['run_id']
        expected = binding['review']['result']
        if expected['device'] != 'cpu':
            raise ValueError(f'{run}: automated tolerances currently cover CPU only; other devices require a recorded tolerance.')
        ws = Workspace.create(destination / run)
        packages.import_package(ws, root / 'run-packages' / f'{run}.zip')
        resolved = experiments.load_resolved(ws, run)
        artifacts = experiments.load_artifacts(ws, run)
        profile = figure['spec']['profiles'][run]
        if resolved.task == TaskType.evaluate_measured:
            signals = measurements.load_signals(ws, run, resolved)
            result = measurements.result_for_measured(ws, run, resolved, artifacts, signals, profile)
            measurements.write_measurement_plots(ws, run, resolved, signals)
        else:
            predictions = evaluation.predict_test_split(ws, run, resolved, artifacts)
            result = evaluation.result_for(ws, run, resolved, artifacts, predictions, profile)
            evaluation.write_plots(ws, run, resolved, predictions)
        row = {'run_id': run, 'metrics': [], 'plots': [], 'passed': True}
        groups = [('primary', expected['metrics'], result.metrics)]
        groups += [(b['kind'], b['metrics'], next((v.metrics for v in result.baselines if v.kind == b['kind']), [])) for b in expected.get('baselines', [])]
        for group, reference, actual in groups:
            for metric in reference:
                value = next((m for m in actual if m.name == metric['name']), None)
                same = bool(value and value.unit == metric['unit'] and value.status.value == metric['status'])
                if same:
                    same = value.value == metric['value'] if value.value is None or metric['value'] is None else math.isclose(value.value, metric['value'], rel_tol=1e-4, abs_tol=1e-5)
                row['metrics'].append({'group': group, 'name': metric['name'], 'expected': metric['value'], 'actual': value.value if value else None, 'passed': same})
                row['passed'] &= same
        for panel in figure['spec']['panels']:
            for trace in panel['traces']:
                if trace['run_id'] != run:
                    continue
                kind = figure_render.plot_kind(panel['kind'])
                key = f'{run}/{kind}'
                original = original_plots[key]
                if kind == 'power_scan':
                    from opendpd.services.figures import _power_plot
                    data = _power_plot(ws, run, profile)
                    # Declared x is retained; the formal y comes from this fresh evaluation.
                    for m in data['traces']:
                        m['y'] = [result.metric(m['name']).value]
                else:
                    data = read_json(ws.run_dir(run) / 'plots' / f'{kind}.json')
                before = next(t for t in original['traces'] if t['name'] == trace['trace_name'])
                after = next(t for t in data['traces'] if t['name'] == trace['trace_name'])
                old_x, old_y, *units = figure_render.coordinates(panel, original, before)
                new_x, new_y, *new_units = figure_render.coordinates(panel, data, after)
                same = (units == new_units and len(old_x) == len(new_x) and len(old_y) == len(new_y)
                        and bool(np.allclose(old_x, new_x, rtol=1e-4, atol=1e-5)) and bool(np.allclose(old_y, new_y, rtol=1e-4, atol=1e-5)))
                exact = old_x == new_x and old_y == new_y
                row['plots'].append({'kind': panel['kind'], 'trace': trace['trace_name'], 'coordinates_exact': exact, 'passed': same})
                row['passed'] &= same
                # Preserve the saved annotations/styles and replace the displayed numerical arrays.
                x_key = 'frequency' if kind == 'spectrum' else 'amp_in' if kind == 'amam' else 'x'
                y_key = 'psd_db' if kind == 'spectrum' else 'amp_out' if panel['kind'] == 'amam' else 'phase_deg' if panel['kind'] == 'ampm' else 'y'
                rebuilt[key][x_key] = new_x
                next(t for t in rebuilt[key]['traces'] if t['name'] == trace['trace_name'])[y_key] = new_y
        report['runs'].append(row)
        report['passed'] &= row['passed']
    figure_render.render_figure(figure, original_plots, root / 'replayed')
    figure_render.render_figure(figure, rebuilt, root / 'reproduced-figures')
    report['png_exact'] = (root / 'figure.png').read_bytes() == (root / 'reproduced-figures' / 'figure.png').read_bytes()
    (root / 'reproduction-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'png_exact': report['png_exact'], 'report': str(root / 'reproduction-check.json')}))
    if not report['passed']:
        raise ValueError('Metric or display-coordinate mismatch; inspect reproduction-check.json.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='.')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--use-bundled-source', action='store_true', help='Execute the hash-verified included OpenDPD source; use only bundles you trust.')
    args = parser.parse_args()
    reproduce(args.directory, args.workspace, args.use_bundled_source)
