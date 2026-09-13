"""Real ILC+ILA run, isolated train/test evidence, and repeatable evaluation."""
import numpy as np
from tests.integration.test_baselines import ws, gru_pa, _run
from opendpd.services.recipes import instantiate, run_dpd_config
from opendpd.services.experiments import load_result, load_artifacts
from opendpd.services.evaluation import evaluate_run


def test_ilc_ila_fit_and_ideal_reference(ws, gru_pa):
    config=instantiate('dpd-ilc-ila-v1','capture',pa_run_id=gru_pa.run_id)
    config.model.parameters={'K':3,'Q':3,'iterations':4,'fit_samples':2048}
    run=_run(ws,config)
    result=load_result(ws,run.run_id)
    assert result.models[0].training_path=='ilc_ila'
    assert result.ilc['test_feedback_used_for_fit'] is False
    assert result.ilc['fit_samples']==2048
    assert result.ilc['test_samples']==result.dataset.n_samples
    assert {b.kind for b in result.baselines}=={'ilc_ideal','surrogate_without_dpd','measured_without_dpd'}
    assert result.ilc['training_history'][-1]['nmse_db'] <= result.ilc['training_history'][0]['nmse_db']
    assert result.ilc['test_history'][-1]['nmse_db'] <= result.ilc['test_history'][0]['nmse_db']
    files={a.artifact_id for a in load_artifacts(ws,run.run_id).artifacts}
    assert {'ilc-json','ilc-ideal-test-csv','ilc-training-npz','ilc-benchmark-json'} <= files
    before=(ws.run_dir(run.run_id)/'ilc-training.npz').read_bytes()
    from opendpd.services.datasets import load_version_arrays
    x, _, split = load_version_arrays(ws, 'capture', result.dataset.preprocessing_version)
    start, _ = split.boundaries['train']
    with np.load(ws.run_dir(run.run_id)/'ilc-training.npz') as saved:
        # Legacy materialization round-trips through decimal CSV; preserve the train prefix within float32 precision.
        np.testing.assert_allclose(saved['x'], x[start:start+2048,0]+1j*x[start:start+2048,1], rtol=1e-7, atol=1e-9)
    again=evaluate_run(ws,run.run_id,result.metric_profile_id)
    assert abs(again.metric('NMSE').value-result.metric('NMSE').value)<1e-6
    applied=_run(ws,run_dpd_config("capture",run.run_id))
    assert load_result(ws,applied.run_id).ilc['fit_samples']==2048
    # Re-evaluation does not alter training signals or fit parameters.
    assert before==(ws.run_dir(run.run_id)/'ilc-training.npz').read_bytes()
