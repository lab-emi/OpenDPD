"""The model registry must match the code and reject bad parameters."""

import pytest

from opendpd.core.registry import RegistryError, get_model, list_models, validate_parameters


def _constructible_legacy_backbones():
    """Legacy CLI choices that models.CoreModel can actually build."""
    from arguments import build_parser
    from models import CoreModel

    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "PA_backbone").choices
    ok = set()
    for name in choices:
        try:
            CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type=name, window_size=4,
                      num_dvr_units=3, thx=0.0, thh=0.0)
            ok.add(name)
        except Exception:
            continue
    return ok


def test_every_registered_model_is_constructible():
    from models import CoreModel
    from opendpd.core.polynomial import PolynomialModel, coefficient_count

    for m in list_models():
        p = m.defaults()
        if m.training_method == "least_squares":
            # fitted in the compute core, never through a legacy backbone
            module = PolynomialModel(m.key, p)
            assert module.n_real_parameters == 2 * coefficient_count(m.key, p)
            continue
        CoreModel(input_size=2, hidden_size=int(p.get("hidden_size", 8)), num_layers=int(p.get("num_layers", 1)),
                  backbone_type=m.legacy_backbone, window_size=4, num_dvr_units=int(p.get("num_dvr_units", 3)),
                  thx=float(p.get("thx", 0.0)), thh=float(p.get("thh", 0.0)))


def test_registry_covers_every_working_legacy_backbone():
    registered = {m.legacy_backbone for m in list_models()}
    missing = _constructible_legacy_backbones() - registered
    assert not missing, f"backbones buildable by models.py but absent from the registry: {sorted(missing)}"


def test_registry_keys_unique_and_have_evidence():
    keys = [m.key for m in list_models()]
    assert len(keys) == len(set(keys))
    for m in list_models():
        assert m.devices_tested, f"{m.key} must list at least one tested device"
        assert m.evidence, f"{m.key} must cite where its test evidence comes from"
        assert m.status in ("supported", "experimental")


def test_unknown_model_is_a_structured_error():
    with pytest.raises(RegistryError) as info:
        get_model("transformer")
    assert info.value.field == "model.key" and "gru" in (info.value.hint or "")


def test_parameters_are_filled_and_checked():
    assert validate_parameters("gru", {}, "pa") == {"hidden_size": 23, "num_layers": 1}
    assert validate_parameters("gru", {"hidden_size": 8}, "dpd")["hidden_size"] == 8
    with pytest.raises(RegistryError) as info:
        validate_parameters("gru", {"hidden": 8}, "pa")
    assert info.value.field == "model.parameters.hidden"
    with pytest.raises(RegistryError):
        validate_parameters("gru", {"hidden_size": 0}, "pa")
    with pytest.raises(RegistryError):
        validate_parameters("gru", {"hidden_size": True}, "pa")
    with pytest.raises(RegistryError):
        validate_parameters("gru", {"hidden_size": 2.5}, "pa")


def test_bojanet_constraint_is_enforced():
    assert validate_parameters("bojanet", {"hidden_size": 18}, "pa")["hidden_size"] == 18
    with pytest.raises(RegistryError):
        validate_parameters("bojanet", {"hidden_size": 19}, "pa")


def test_lookahead_is_declared_or_explicitly_unknown():
    for m in list_models():
        assert m.lookahead_note
        if m.lookahead_samples is None:
            assert "not characterised" in m.lookahead_note
    assert get_model("tres_deltagru").lookahead_samples == 16
    assert get_model("tcn").lookahead_samples == 30
    assert get_model("gru").lookahead_samples == 0
