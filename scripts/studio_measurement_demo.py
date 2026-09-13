"""Add explicitly MOCK integer/fractional capture reviews to an existing research demo.

Uses existing DPD weights, a synthetic delayed complex gain plus noise, and the
normal apply/measurement pipelines. No instrument or RF output is accessed.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opendpd.schemas import CaptureRef, MeasurementConditions, RunStatus
from opendpd.schemas.measurement_session import MeasurementSessionSpec, SessionCapture
from opendpd.services import experiments, measurements, measurement_sessions
from opendpd.services.recipes import run_dpd_config
from opendpd.services.workspace import Workspace, read_json, write_json_atomic


def build(root):
    ws = Workspace.open_or_create(root)
    ws.imports_dir.mkdir(exist_ok=True)
    demo = read_json(ws.root / 'review-demo' / 'index.json')
    ids = demo['runs']
    cfg = run_dpd_config('dpa-200mhz', ids['gru'], pa_run_id=ids['pa'])
    apply = experiments.create_run(ws, cfg, idempotency_key='review-mock-apply-v2')
    if apply.status == RunStatus.queued:
        apply = experiments.execute_run(ws, apply.run_id)
    assert apply.status == RunStatus.succeeded, apply.error
    artifact = next(a for a in experiments.load_artifacts(ws, apply.run_id).artifacts if a.kind.value == 'dpd_output')
    x, u = measurements.read_played(ws.run_dir(apply.run_id) / artifact.file.path)
    capture_refs, runs = [], []
    for i in range(2):
        rng = np.random.default_rng(500 + i)
        capture_path = ws.imports_dir / f'review-MOCK-acquisition-{i}.npy'
        delayed = np.fft.ifft(np.fft.fft(u) * np.exp(-2j * np.pi * np.fft.fftfreq(len(u)) * 123.37))
        y = 3 * np.exp(.4j) * np.tile(delayed, 2)
        y += .0001 * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
        np.save(capture_path, y)
        time = datetime(2026, 9, 13, 12, tzinfo=timezone.utc) + timedelta(minutes=i)
        conditions = MeasurementConditions(pa='MOCK delayed complex gain', capture_chain='synthetic: gain 3, phase 0.4 rad, delay 123.37 samples, independent noise seed', sample_rate_hz=800e6, drive='synthetic input; no RF output', measured_at=time, calibration='none; synthetic demonstration')
        for version in (['measurement-integer-v1', 'measurement-fractional-v2'] if i == 0 else ['measurement-fractional-v2']):
            config = measurements.measurement_config(ws, apply.run_id, with_dpd=CaptureRef(path=capture_path.name, declared_output_power_dbm=30), without_dpd=None, conditions=conditions, source='mock_adapter', processing_version=version, profile_id='general-spectral-v1', name=f'MOCK acquisition {i}: {version}')
            run = experiments.create_run(ws, config, idempotency_key=f'review-MOCK-{i}-{version}')
            if run.status == RunStatus.queued:
                run = experiments.execute_run(ws, run.run_id)
            assert run.status == RunStatus.succeeded, run.error
            runs.append(run.run_id)
            if version.endswith('v2'):
                capture_refs.append(SessionCapture(capture_id=f'mock-capture-{i}', acquisition_id=f'synthetic-acquisition-{i}', run_id=run.run_id, role='with_dpd', acquired_at=time, raw_units='synthetic arbitrary units'))
    session = measurement_sessions.create_session(ws, MeasurementSessionSpec(title='MOCK acquisition grouping and fractional alignment', dut=conditions.pa, source='mock', profile_id='general-spectral-v1', captures=capture_refs, power_tolerance_db=.1))
    index = {'evidence': 'MOCK only: synthetic captures; no real repeatability or calibration validation', 'runs': runs, 'session_id': session.session_id, 'result_path': f'/results/{runs[1]}?profile=general-spectral-v1', 'known': {'delay_samples': 123.37, 'gain': 3, 'phase_rad': .4}}
    write_json_atomic(ws.root / 'review-demo' / 'measurement-index.json', index)
    print(json.dumps(index, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    build(parser.parse_args().workspace)
