"""The cost of judging untrusted recipes: one plan per canonical recipe, no NB-IoT frame during validation, a CPU budget for the worst bodies.

The public Studio validates request bodies of untrusted size on a shared process. A body can carry 16 configurations of 16 carriers each
and every cell identity changes the canonical recipe, so the work must not depend on a cache being warm: the plan of a recipe is a few
arithmetic operations (about a millisecond), and the NB-IoT frames, whose Gold sequences cost ten milliseconds each, are built only when a
signal is synthesised.
"""
import copy
import json
import random
import threading
import time

import pytest
from pydantic import ValidationError

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm import nb_iot, plan as plan_module
from opendpd.core.waveforms.nr_tm.tables import NB_IOT_BANDWIDTHS_MHZ
from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier, NrTmNbIot
from opendpd.schemas.signal_generator import GeneratorBatchRequest, GeneratorConfig


def recipe(model="NR-FR1-TM1.1", *cells, **kwargs):
    carriers = [NrTmCarrier(scs_khz=scs, bandwidth_mhz=bw, n_cell_id=cell) for scs, bw, cell in (cells or [(30, 20, None)])]
    return NrTestModelConfig(model=model, carriers=carriers, **kwargs)


def ntm(host_ids, bandwidth=50, position="both", boost=6.0, carriers=1, **kwargs):
    """NR-N-TM with `carriers` carriers; the lowest and the highest carry the given cell identities (they decide the NB-IoT N_IDs)."""
    cells = [NrTmCarrier(scs_khz=15, bandwidth_mhz=bandwidth, n_cell_id=host_ids[0])]
    cells += [NrTmCarrier(scs_khz=15, bandwidth_mhz=bandwidth) for _ in range(carriers - 2)]
    if carriers > 1:
        cells.append(NrTmCarrier(scs_khz=15, bandwidth_mhz=bandwidth, n_cell_id=host_ids[-1]))
    return NrTestModelConfig(model="NR-N-TM", carriers=cells, nb_iot=NrTmNbIot(position=position, boost_db=boost), **kwargs)


@pytest.fixture(autouse=True)
def cold_memo():
    nr_tm.clear_memo()
    yield
    nr_tm.clear_memo()


@pytest.fixture
def plans(monkeypatch):
    """Counts the plans that are really computed (the memo hides the others)."""
    calls = []
    real = plan_module.make_plan

    def counting(r):
        calls.append(r.model)
        return real(r)
    monkeypatch.setattr(plan_module, "make_plan", counting)
    return calls


# ---------------------------------------------------------------------------------------------------------------------
# the memo
# ---------------------------------------------------------------------------------------------------------------------

def test_a_recipe_is_planned_once_for_every_caller(plans):
    r = recipe("NR-FR1-TM3.1a", (30, 20, None), (30, 20, None))
    assert len(plans) == 1                                           # the recipe validator
    nr_tm.validate_recipe(r)
    nr_tm.derive(r)
    nr_tm.derived_fields(r)
    resolution = nr_tm.resolve(r)                                    # derive + GeneratorConfig._match_test_model
    GeneratorConfig.model_validate(resolution.config.model_dump(mode="json"))
    GeneratorBatchRequest.model_validate({"configs": [resolution.config.model_dump(mode="json")]})
    assert nr_tm.resolve(NrTestModelConfig.model_validate(r.model_dump(mode="json"))).config == resolution.config
    assert len(plans) == 1
    recipe("NR-FR1-TM3.1a", (30, 20, None), (30, 20, 5))             # another canonical recipe is another plan
    assert len(plans) == 2


def test_equal_recipes_share_one_entry_however_they_were_written(plans):
    a = NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 6}, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}]})
    b = NrTestModelConfig.model_validate({"carriers": [{"bandwidth_mhz": 10, "scs_khz": 15, "gap_before_mhz": -0.0, "n_cell_id": None}],
                                          "nb_iot": {"boost_db": 6.0, "position": "both"}, "model": "NR-N-TM"})
    assert a == b and len(plans) == 1


def test_a_refusal_is_memoised_with_its_message(plans):
    refused = NrTestModelConfig.model_construct(model="NR-FR1-TM1.1", carriers=[NrTmCarrier(scs_khz=60, bandwidth_mhz=5)])
    messages = set()
    for _ in range(3):
        with pytest.raises(ValueError, match="5 MHz channel is not defined for 60 kHz") as caught:
            nr_tm.validate_recipe(refused)
        messages.add(str(caught.value))
    assert len(messages) == 1 and len(plans) == 1
    with pytest.raises(ValueError, match="5 MHz channel is not defined"):
        nr_tm.derive(refused)
    assert len(plans) == 1


def test_the_memo_is_bounded_and_forgets_the_least_recently_used(monkeypatch):
    monkeypatch.setattr(plan_module, "MEMO_SIZE", 4)
    recipes = [recipe("NR-FR1-TM1.1", (30, 20, cell)) for cell in range(10, 16)]
    keys = [plan_module.canonical_json(r) for r in recipes]
    assert len(plan_module._memo) == 4 and list(plan_module._memo) == keys[2:]
    nr_tm.validate_recipe(recipes[2])                                # a hit refreshes the entry
    assert list(plan_module._memo)[-1] == keys[2]
    recipe("NR-FR1-TM1.1", (30, 20, 99))
    assert keys[2] in plan_module._memo and keys[3] not in plan_module._memo and len(plan_module._memo) == 4


def test_callers_cannot_corrupt_what_the_memo_hands_to_the_next_caller():
    r = recipe("NR-FR1-TM3.2", (30, 20, None), (30, 20, None))
    reference = plan_module.make_plan(recipe("NR-FR1-TM3.2", (30, 20, None), (30, 20, None)))     # uncached, with a recipe nobody touches
    expected_fields = plan_module.structural_fields(reference)
    expected_summary = plan_module.summarize(reference)
    first = nr_tm.derive(r)
    first.fields["n_samples"] = 1
    first.fields["channel_subcarriers"].append(7)
    first.summary.carriers[0].nrb = 1
    first.summary.carriers[0].boosted_rbgs.append(99)
    first.summary.warnings.append("tampered")
    first.summary.carriers.pop()
    nr_tm.derived_fields(r)["channel_modulations"].clear()
    r.carriers[0].n_cell_id = 77                                       # the caller changes its own recipe afterwards
    again = nr_tm.derive(recipe("NR-FR1-TM3.2", (30, 20, None), (30, 20, None)))
    assert again.fields == expected_fields and again.summary == expected_summary
    assert nr_tm.derived_fields(recipe("NR-FR1-TM3.2", (30, 20, None), (30, 20, None))) == expected_fields
    assert again.plan.recipe.carriers[0].n_cell_id is None and again.plan == reference


def test_validation_is_thread_safe_and_gives_every_thread_the_sequential_answer():
    recipes = [ntm((cell, cell + 9), bandwidth=10, carriers=3, boost=7.0) for cell in range(1, 21)]
    expected = {plan_module.canonical_json(r): plan_module.structural_fields(plan_module.make_plan(r)) for r in recipes}
    nr_tm.clear_memo()
    failures = []

    def work(seed):
        rnd = random.Random(seed)
        try:
            for _ in range(60):
                r = rnd.choice(recipes)
                assert nr_tm.derived_fields(r) == expected[plan_module.canonical_json(r)]
                assert nr_tm.derive(r).summary.carriers[0].n_cell_id == r.carriers[0].n_cell_id
        except Exception as exc:                                       # noqa: BLE001 - reported below
            failures.append(repr(exc))
    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures, failures[:3]


def test_assigning_an_editable_field_does_not_derive_the_recipe_again(plans):
    config = nr_tm.resolve(recipe("NR-FR1-TM1.1", (30, 20, None))).config
    nr_tm.clear_memo()
    plans.clear()
    config.rms = 0.3
    config.seed = 5
    config.snr_db = 40.0
    assert plans == []
    with pytest.raises(ValidationError, match="derived from its nr_test_model recipe"):
        config.fft_size = 4096                                         # a structural field is still checked
    assert len(plans) == 1


# ---------------------------------------------------------------------------------------------------------------------
# NB-IoT feasibility in closed form
# ---------------------------------------------------------------------------------------------------------------------

def test_validation_and_resolution_build_no_nb_iot_frame(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an NB-IoT frame was built while validating")
    monkeypatch.setattr(nb_iot, "_build_frame", forbidden)
    monkeypatch.setattr(nb_iot, "nb_iot_frame", forbidden)
    for bw in NB_IOT_BANDWIDTHS_MHZ:
        for position in ("lower", "upper", "both"):
            r = ntm((3, 390), bandwidth=bw, position=position, boost=6.0, carriers=2)
            resolution = nr_tm.resolve(r)
            assert resolution.summary.nb_iot_rbs
    with pytest.raises(ValidationError, match="boost of 12 dB is not possible"):
        ntm((1, 2), bandwidth=3, position="both", boost=12.0)
    with pytest.raises(ValidationError, match="exceeds 503"):
        ntm((450, 460), bandwidth=10, carriers=2)
    plan = plan_module.make_plan(ntm((1, 2), bandwidth=10, carriers=2))
    with pytest.raises(AssertionError, match="while validating"):
        plan.nb_iot_overlays                                           # the frames come only with the signal


def test_the_closed_form_budget_equals_the_frames_it_stands_for():
    for host in range(6):
        for n_id in [0, 1, 2, 3, 4, 5, 7, 103, 110, 308, 503]:
            counted = nb_iot.frame_re_count(n_id, host_cell_id=host)
            assert counted == int(nb_iot.nb_iot_frame(n_id, host_cell_id=host)[1].sum()), (n_id, host)
    for kwargs in ({"mode": "standalone"}, {"nrs_ports": 1}, {"mode": "standalone", "nrs_ports": 1}, {"control_symbols": 2}, {"host_crs_ports": 2}):
        assert nb_iot.frame_re_count(103, **kwargs) == int(nb_iot.nb_iot_frame(103, **kwargs)[1].sum()), kwargs
    assert nb_iot.frame_re_count(103) == 1085 and nb_iot.frame_re_count(110, host_cell_id=2) == 1089     # host cell identity 2 mod 6: the NPSS skips no CRS
    with pytest.raises(ValueError, match="N_ID"):
        nb_iot.frame_re_count(504)


def test_plans_agree_with_the_overlays_built_for_synthesis():
    for bw in (3, 10, 20, 50):
        for position in ("lower", "upper", "both"):
            plan = plan_module.make_plan(ntm((2, 6), bandwidth=bw, position=position, boost=6.5, carriers=3))
            overlays = dict(plan.nb_iot_overlays)
            assert sorted(overlays) == [h.carrier for h in plan.nb_iot_hosts]
            for host in plan.nb_iot_hosts:
                overlay = overlays[host.carrier]
                assert (overlay.rbs, overlay.n_nb_re, overlay.nr_gain, overlay.nb_gain) == (list(host.rbs), host.n_nb_re, host.nr_gain, host.nb_gain)
                assert {c.n_id for c in overlay.carriers} == {host.n_id}


def test_the_nb_iot_layout_is_shared_by_all_cell_identities_of_one_residue():
    nb_iot._layout.cache_clear()
    nb_iot.frame_re_count(7)
    nb_iot.frame_re_count(13)
    nb_iot.frame_re_count(499)
    info = nb_iot._layout.cache_info()
    assert (info.misses, info.hits) == (1, 2)                         # N_ID 7, 13 and 499 are all 1 modulo 6
    a, b = nb_iot.nb_iot_frame(7), nb_iot.nb_iot_frame(13)
    assert (a[1] == b[1]).all() and not (a[0] == b[0]).all()           # same positions, different Gold sequences


# ---------------------------------------------------------------------------------------------------------------------
# the worst bodies
# ---------------------------------------------------------------------------------------------------------------------
# CPU time of the validating thread (time.thread_time), the best of three cold runs. process_time would also count native threads that
# have nothing to do with validation (OpenBLAS threads of numpy spin for hundreds of milliseconds after an import). The old code needed
# 190-370 ms for these bodies on the machine that wrote this test; the budgets below leave a factor of 5-8 over the new cost and stay far
# below the old one, so a regression to per-config frame building (or to one derivation per validator) fails them.

def cpu_ms(work, repeats=3):
    best = float("inf")
    for attempt in range(repeats):
        started = time.thread_time()
        work(attempt)
        best = min(best, (time.thread_time() - started) * 1000)
    return best


def batch_of_wide_ntm_configs(attempt):
    """16 valid configurations of 16-carrier NR-N-TM (50 MHz hosts, two NB-IoT carriers each) with cell identities that no earlier run used."""
    configs = []
    for i in range(16):
        base = 1 + attempt * 60 + i * 3                              # the second host's N_ID must stay at most 503: base + 190 <= 395
        configs.append(nr_tm.resolve(ntm((base, base + 190), bandwidth=50, carriers=16, subframes=1)).config.model_dump(mode="json"))
    return configs


def test_the_worst_batch_body_is_judged_within_a_small_cpu_budget():
    bodies = [batch_of_wide_ntm_configs(attempt) for attempt in range(3)]
    nr_tm.clear_memo()
    assert 30_000 < len(json.dumps(bodies[0])) < 65_536                # a real 16 x 16-carrier body that the public body limit lets through

    def validate(attempt):
        with pytest.raises(ValidationError, match="at most 4,000,000 samples"):          # legal configurations, refused after all 16 were judged
            GeneratorBatchRequest.model_validate({"configs": bodies[attempt]})
    assert cpu_ms(validate) < 50
    # judging the same body again is a lookup per configuration
    started = time.thread_time()
    validate(0)
    assert (time.thread_time() - started) * 1000 < 25


def test_late_failing_single_recipes_cost_a_fraction_of_a_millisecond():
    """Boost 12 dB with position 'both' on a 3 MHz host is refused by the power budget, the last thing the plan checks."""
    def validate(attempt):
        for cell in range(1 + 100 * attempt, 101 + 100 * attempt):
            with pytest.raises(ValidationError, match="boost of 12 dB is not possible"):
                NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 12},
                                                  "carriers": [{"scs_khz": 15, "bandwidth_mhz": 3, "n_cell_id": cell}]})
    assert cpu_ms(validate) < 150                                      # 100 distinct recipes; the old code needed about 12 ms each


def test_a_sixteen_carrier_recipe_costs_a_few_milliseconds_cold():
    recipes = [{"model": "NR-FR1-TM3.1a", "duplex": "tdd", "carriers": [{"scs_khz": 30, "bandwidth_mhz": 100 if i % 2 else 10, "n_cell_id": 1 + attempt * 20 + i}
                                                                       for i in range(16)]} for attempt in range(3)]

    def validate(attempt):
        with pytest.raises(ValidationError):
            NrTestModelConfig.model_validate(recipes[attempt])
    assert cpu_ms(validate) < 50


def test_every_edit_of_a_valid_body_stays_cheap_after_the_first_judgement():
    base = batch_of_wide_ntm_configs(0)[0]
    nr_tm.clear_memo()
    GeneratorConfig.model_validate(base)
    started = time.thread_time()
    for _ in range(50):
        config = copy.deepcopy(base)
        config["rms"] = 0.3
        GeneratorConfig.model_validate(config)
    assert (time.thread_time() - started) * 1000 < 150


def test_the_validate_route_judges_a_cold_sixteen_carrier_config_within_a_few_milliseconds():
    """POST /signal-generator/validate has no recipe budget of its own (the decision is pinned in tests/unit/test_public_nr_tm_policy.py).
    What one request costs, parsed, validated, prechecked and serialised the way the route does it, for the widest recipe (16 carriers of
    50 MHz NR-N-TM with two NB-IoT carriers on each end) that nobody has judged before: about 1.5 ms of thread time on the machine that wrote
    this test (0.6 ms of it on the event loop, in the body validation); the release before the plan memo needed about 6.7 ms. The budget is
    the best of three cold runs, 5 ms per request."""
    from opendpd.services import signal_generator as service

    bodies = [[json.dumps(config).encode() for config in batch_of_wide_ntm_configs(attempt)] for attempt in range(3)]
    nr_tm.clear_memo()                                               # every recipe is judged for the first time, as an attacker's would be

    def route(attempt):
        for payload in bodies[attempt]:
            config = GeneratorConfig.model_validate_json(payload)
            service.precheck(config)
            config.model_dump(mode="json")
    assert cpu_ms(route) / len(bodies[0]) < 5
