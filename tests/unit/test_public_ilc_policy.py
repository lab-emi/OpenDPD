"""The public ILC recipe is usable and bounded before a worker is admitted."""
import pytest
from fastapi import HTTPException
from opendpd.services.recipes import instantiate
from opendpd.web.policy import check_config, allowed


def test_public_ilc_defaults_and_input_removal_routes():
    config = instantiate('dpd-ilc-ila-v1', 'capture', pa_run_id='pa-fixture')
    check_config(config)
    for action in ('archive', 'restore'):
        assert allowed('POST', '/signal-generator/signals/sg-' + 'a' * 64 + '/' + action)
        assert not allowed('POST', '/signal-generator/signals/../' + action)


@pytest.mark.parametrize('key,value', [('K',11),('Q',17),('iterations',61),('fit_samples',32769),('backtracking_steps',7)])
def test_public_ilc_compute_caps(key, value):
    config = instantiate('dpd-ilc-ila-v1', 'capture', pa_run_id='pa-fixture')
    config.model.parameters[key] = value
    with pytest.raises(HTTPException) as error:
        check_config(config)
    assert error.value.status_code == 422
    assert error.value.detail['error']['code'] == 'compute_limit'
