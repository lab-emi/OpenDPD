"""Hostile numbers in NR test-model recipes: every validator ends in a validation error and the public boundary answers 422, never 500.

A pydantic ``mode="before"`` validator sees the raw value of a request body, so anything JSON can say (a 400-digit integer, ``1e999``,
``Infinity``, a list where a number belongs) reaches it. Pydantic turns ``ValueError``, ``AssertionError`` and its own errors into a
validation error; every other exception escapes as a 500 (and, on the public Studio, a traceback in the log per request).
"""
import asyncio
import logging
import math
import random
import re
from decimal import Decimal
from fractions import Fraction

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm import nb_iot
from opendpd.schemas.nr_test_model import MAX_NB_IOT_BOOST_DB, NrTestModelConfig, NrTmNbIot, shortest_number
from opendpd.schemas.signal_generator import GeneratorBatchRequest, GeneratorConfig
from opendpd.web import mutation_policy

# (value, text pydantic or the validator must show). Values that a client can really send are JSON numbers, strings, null, booleans,
# arrays and objects; the Python-only ones (Decimal, Fraction, bytes, ...) are what other callers of the schemas can pass.
HOSTILE_BOOST = [
    (10 ** 400, "valid number"), (-10 ** 400, "valid number"), (2 ** 1100, "valid number"), (10 ** 4299, "valid number"),
    ("1e999", "finite number"), ("-1e999", "finite number"), ("Infinity", "finite number"), ("nan", "finite number"), ("9" * 5000, "finite number"),
    (1e999, "finite number"), (-1e999, "finite number"), (float("nan"), "finite number"),
    (True, "greater than or equal to 3"), (False, "greater than or equal to 3"), (-0.0, "greater than or equal to 3"), (0, "greater than or equal to 3"),
    (None, "valid number"), ([], "valid number"), ({}, "valid number"), ([13], "valid number"), ("x", "valid number"), ("", "valid number"),
    ([[[[[1]]]]], "valid number"), (object(), "valid number"), (1 + 2j, "valid number"),
    (Decimal("NaN"), "finite number"), (Decimal("sNaN"), "valid number"), (Decimal("-Infinity"), "finite number"), (Decimal("1e999"), "finite number"),
    (Fraction(10 ** 400, 1), "valid number"), (b"13", "less than or equal to 12"), (2.9999999, "greater than or equal to 3"),
]

NB_IOT_RECIPE = {"model": "NR-N-TM", "nb_iot": {"position": "lower", "boost_db": 6}, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}]}


UNSET = object()


def recipe_with(boost=UNSET, gap=UNSET):
    """An NR-N-TM recipe of two carriers whose boost and second gap are the given raw values (None is a value, not 'unset')."""
    return {"model": "NR-N-TM", "nb_iot": {"position": "lower", "boost_db": 6 if boost is UNSET else boost},
            "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}, {"scs_khz": 15, "bandwidth_mhz": 10, "gap_before_mhz": 0 if gap is UNSET else gap}]}


def refusals(body, stored=True):
    """Judge `body` as a recipe, inside a configuration and inside a batch (and, with `stored`, as a record read from storage, which only has
    to be well-formed): every judgement must be a validation error."""
    shapes = [lambda: NrTestModelConfig.model_validate(body), lambda: GeneratorConfig.model_validate({"nr_test_model": body}),
              lambda: GeneratorBatchRequest.model_validate({"configs": [{"nr_test_model": body}]})]
    if stored:
        shapes += [lambda: NrTestModelConfig.model_validate(body, context={"stored": True}),
                   lambda: GeneratorConfig.model_validate({"nr_test_model": body}, context={"stored": True})]
    errors = []
    for shape in shapes:
        with pytest.raises(ValidationError) as caught:
            shape()
        errors.append(caught.value)
    return errors


@pytest.mark.parametrize("value,text", HOSTILE_BOOST, ids=lambda v: type(v).__name__ if not isinstance(v, str) else v[:12])
def test_a_hostile_boost_is_a_validation_error_whatever_its_type_or_size(value, text):
    with pytest.raises(ValidationError, match=text):
        NrTmNbIot.model_validate({"position": "lower", "boost_db": value})
    for error in refusals(recipe_with(boost=value)):
        assert any(tuple(e["loc"])[-1] == "boost_db" for e in error.errors()), error.errors()


def test_a_huge_integer_is_answered_like_any_other_malformed_number():
    for value in (10 ** 400, -10 ** 400, 2 ** 1100):
        with pytest.raises(ValidationError) as caught:
            NrTmNbIot.model_validate({"position": "lower", "boost_db": value})
        assert [e["type"] for e in caught.value.errors()] in (["float_parsing"], ["float_type"], ["finite_number"], ["greater_than_equal"], ["less_than_equal"])


@pytest.mark.parametrize("value,shown", [(12.0000001, "12.0000001"), (12.000000000000002, "12.000000000000002"), (12.5, "12.5"), (13, "13"), (13.0, "13"),
                                         ("13", "13"), (" 13 ", "13"), (1e22, "1e+22"), (1e300, "1e+300"), (Decimal("13.25"), "13.25")])
def test_a_boost_above_the_cap_prints_the_offending_value_so_that_the_text_never_contradicts_itself(value, shown):
    with pytest.raises(ValidationError) as caught:
        NrTmNbIot.model_validate({"position": "lower", "boost_db": value})
    message = caught.value.errors()[0]["msg"]
    assert f"{shown} dB is above the largest NB-IoT boost Studio offers ({MAX_NB_IOT_BOOST_DB} dB," in message
    assert shown != str(MAX_NB_IOT_BOOST_DB)


@pytest.mark.parametrize("value", [3, 3.0, "3", 6, "6.5", 11.999999999999998, 12, 12.0, "12", Decimal("12"), Decimal("7.5")])
def test_boosts_up_to_the_cap_are_not_this_validators_business(value):
    assert NrTmNbIot.model_validate({"position": "lower", "boost_db": value}).boost_db == float(value)


@pytest.mark.parametrize("number,text", [(13.0, "13"), (12.0000001, "12.0000001"), (0.1 + 0.2, "0.30000000000000004"), (1e16, "1e+16"), (-2.5, "-2.5"), (7, "7")])
def test_shortest_number_reads_back_as_the_number_without_a_trailing_zero(number, text):
    assert shortest_number(number) == text and float(shortest_number(number)) == float(number)


# ---- the sub-block gap ---------------------------------------------------------------------------------------------

HOSTILE_GAP = [10 ** 400, -10 ** 400, 2 ** 1100, "1e999", "-1e999", "nan", 1e999, -1e999, float("nan"), None, [], {}, "x", [[[1]]], Decimal("NaN"), Decimal("1e999"),
               1000.0000001, -1e-9, -0.3]
OFF_THE_GRID = [1e-8, 0.30000001, 0.3 + 2e-9, 5.5, 299.99, "0.31", Decimal("0.35")]


@pytest.mark.parametrize("gap", HOSTILE_GAP, ids=lambda v: type(v).__name__ + ":" + str(v)[:10])
def test_a_hostile_gap_is_a_validation_error_for_every_caller_including_storage(gap):
    for error in refusals(recipe_with(gap=gap)):
        assert any(tuple(e["loc"])[-1] == "gap_before_mhz" for e in error.errors()), error.errors()


@pytest.mark.parametrize("gap", OFF_THE_GRID, ids=str)
def test_a_gap_off_the_grid_is_refused_to_a_user_and_tolerated_in_a_stored_record(gap):
    for error in refusals(recipe_with(gap=gap), stored=False):
        assert "must be a multiple of 0.3 MHz" in error.errors()[0]["msg"] or "gap_before_mhz" in str(error.errors()[0]["loc"])
    stored = NrTestModelConfig.model_validate(recipe_with(gap=gap), context={"stored": True})
    assert stored.carriers[1].gap_before_mhz == float(gap)


@pytest.mark.parametrize("gap,stored", [(-0.0, 0.0), (0.8999999999999999, 0.9), (0.30000000000000004, 0.3), (0.3 + 1e-10, 0.3), ("0.6", 0.6), (99.9, 99.9), (Decimal("1.2"), 1.2)])
def test_a_gap_on_the_grid_is_stored_as_its_exact_multiple(gap, stored):
    carriers = NrTestModelConfig.model_validate(recipe_with(gap=gap)).carriers
    assert carriers[1].gap_before_mhz == stored and math.copysign(1, carriers[1].gap_before_mhz) == 1
    assert NrTestModelConfig.model_validate(recipe_with(gap=gap), context={"stored": True}).carriers[1].gap_before_mhz in (stored, float(gap))


# ---- a seeded sweep through the whole schema ------------------------------------------------------------------------

POOL = [10 ** 400, -10 ** 400, 2 ** 1100, "1e999", "nan", "9" * 5000, 1e999, -1e999, float("nan"), 1e308, 5e-324, -0.0, 0.3, 0.6, 0.8999999999999999, 12.0000001, 12, 13,
        5.9999999, 3, 6, 1000, 1000.0000001, None, True, [], {}, [1], [[[[1]]]], "x", "", " 6 ", "0.3", Decimal("NaN"), Decimal("sNaN"), Decimal("6.5"), Fraction(1, 3),
        0, 1, -1, 503, 504, 1007, 1008, 10.0, "10", 15, 30, 60, "fdd", "tdd", "lower", "upper", "both"]
BASES = [
    NB_IOT_RECIPE,
    {"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 6}, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10, "n_cell_id": 1}, {"scs_khz": 15, "bandwidth_mhz": 20, "gap_before_mhz": 0.3}]},
    {"model": "NR-FR1-TM3.2", "duplex": "tdd", "data_convention": "matlab", "pn_scope": "all_symbols", "oversampling": 2, "subframes": 3,
     "carriers": [{"scs_khz": 30, "bandwidth_mhz": 40, "n_cell_id": 7}, {"scs_khz": 60, "bandwidth_mhz": 40, "n_cell_id": 8, "gap_before_mhz": 0.6}]},
    {"model": "NR-FR1-TM1.1", "layers": 2, "output_port": 1, "subframes": 1, "oversampling": 1, "carriers": [{"scs_khz": 30, "bandwidth_mhz": 20}]},
]


def leaves(obj, prefix=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, prefix + (key,))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, prefix + (i,))
    else:
        yield prefix


def with_value(obj, path, value):
    import copy
    obj = copy.deepcopy(obj)
    parent = obj
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return obj


def test_nothing_but_a_validation_error_comes_out_of_a_seeded_sweep_of_hostile_values():
    rnd = random.Random(20260511)
    outcomes = {"valid": 0, "refused": 0}
    for _ in range(400):
        body = rnd.choice(BASES)
        for _ in range(rnd.randint(1, 3)):
            body = with_value(body, rnd.choice([p for p in leaves(body) if p]), rnd.choice(POOL))
        for judge in (lambda: NrTestModelConfig.model_validate(body), lambda: NrTestModelConfig.model_validate(body, context={"stored": True}),
                      lambda: GeneratorConfig.model_validate({"nr_test_model": body}), lambda: GeneratorBatchRequest.model_validate({"configs": [{"nr_test_model": body}]}),
                      lambda: nr_tm.resolve(NrTestModelConfig.model_validate(body))):
            try:
                judge()
                outcomes["valid"] += 1
            except ValidationError:
                outcomes["refused"] += 1
    assert outcomes["valid"] > 20 and outcomes["refused"] > 1000         # the sweep reaches the plan as well as the field checks


def test_deeply_nested_values_are_validation_errors():
    deep = []
    for _ in range(3000):
        deep = [deep]
    for body in ({"model": "NR-N-TM", "carriers": deep}, {"model": "NR-N-TM", "carriers": [deep]}, {"model": "NR-N-TM", "nb_iot": deep, "carriers": [{}]},
                 {"model": "NR-N-TM", "nb_iot": {"boost_db": deep}, "carriers": [{"gap_before_mhz": deep}]}):
        refusals(body)


# ---- the plan's messages about the boost -----------------------------------------------------------------------------

def test_a_boost_below_the_minimum_of_the_host_prints_the_value_it_was_given():
    for boost in (5.9999999, 5.999999999999999, 3.0000001):
        with pytest.raises(ValidationError) as caught:
            NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "lower", "boost_db": boost}, "carriers": [{"scs_khz": 15, "bandwidth_mhz": 10}]})
        message = caught.value.errors()[0]["msg"]
        assert "must be at least 6 dB" in message and f"got {shortest_number(boost)} dB" in message


@pytest.mark.parametrize("cell", range(1, 13))
def test_a_refused_boost_never_reads_smaller_than_the_limit_that_refuses_it(cell):
    """Only a 3 MHz host with two NB-IoT carriers cannot take 12 dB (its limit is about 10.6 dB and moves with the cell identity).
    Bisect to the smallest boost the power budget refuses: the printed boost is at least the printed limit, which is rounded down."""
    def refusal(boost):
        try:
            nr_tm.resolve(NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": boost},
                                                            "carriers": [{"scs_khz": 15, "bandwidth_mhz": 3, "n_cell_id": cell}]}))
        except ValidationError as error:
            return error.errors()[0]["msg"]
    low, high = 6.0, float(MAX_NB_IOT_BOOST_DB)
    assert refusal(low) is None and refusal(high) is not None
    while high - low > 1e-12:
        middle = (low + high) / 2
        low, high = (middle, high) if refusal(middle) is None else (low, middle)
    message = refusal(high)
    boost, limit = (float(x) for x in re.search(r"boost of ([0-9.e+-]+) dB is not possible.*stay below ([0-9.]+) dB", message).groups())
    assert boost == high and boost - 0.01 < limit <= boost                 # the limit is rounded down to the cent, never up


@pytest.mark.parametrize("cell,boost,shown", [(1, 11, "10.65"), (1, 10.65, "10.64"), (1, 10.6495, "10.64"), (1, 10.6494081, "10.64"), (2, 10.64, "10.63"), (2, 11, "10.63")])
def test_the_limit_is_printed_to_the_nearest_cent_unless_that_would_exceed_the_boost_it_refuses(cell, boost, shown):
    """The true limit of two NB-IoT carriers on a 3 MHz host is 10.6494 dB (cell 1) or 10.6334 dB (cell 2)."""
    with pytest.raises(ValidationError) as caught:
        NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": boost},
                                          "carriers": [{"scs_khz": 15, "bandwidth_mhz": 3, "n_cell_id": cell}]})
    message = caught.value.errors()[0]["msg"]
    assert f"boost of {shortest_number(boost)} dB is not possible" in message and f"must stay below {shown} dB to leave power" in message
    assert float(shown) <= boost


def test_the_single_carrier_hint_never_offers_more_than_the_power_budget_or_studio_allows():
    """One NB-IoT carrier on a 3 MHz host could take 13.6 dB, but Studio stops at 12 dB: the hint must offer what the form accepts."""
    carriers = [{"scs_khz": 15, "bandwidth_mhz": 3, "n_cell_id": 1}]
    with pytest.raises(ValidationError) as caught:
        NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "both", "boost_db": 11}, "carriers": carriers})
    offered = float(re.search(r"up to ([0-9.]+) dB\)", caught.value.errors()[0]["msg"]).group(1))
    assert nb_iot.max_boost_db(15, nb_iot.frame_re_count(nb_iot.nb_iot_n_id(1, 1), host_cell_id=1)) > 13        # the power budget alone would allow more
    assert offered == MAX_NB_IOT_BOOST_DB
    NrTestModelConfig.model_validate({"model": "NR-N-TM", "nb_iot": {"position": "lower", "boost_db": offered}, "carriers": carriers})


# ---- the public boundary --------------------------------------------------------------------------------------------

class Manager:
    def __init__(self):
        self.limits = []

    def rate_limit(self, key, limit=None):
        self.limits.append((key, limit))


class Tenant:
    ip_key = "ip"
    root = None


def budget(path, body):
    manager = Manager()
    try:
        asyncio.run(mutation_policy.enforce_mutation_budget(manager, Tenant(), path, body))
    except HTTPException as refused:
        return manager, refused
    return manager, None


@pytest.fixture
def reports(monkeypatch):
    monkeypatch.setattr(mutation_policy, "_REPORTED", set())


RULES = {"/signal-generator/nr-test-models/resolve": 1, "/signal-generator/batches": 1, "/signal-generator/signals": 1, "/pa-library/simulations": 0}


@pytest.mark.parametrize("failure", [OverflowError("int too large to convert to float: SECRET-BODY"), RecursionError("maximum recursion depth: SECRET-BODY"),
                                     MemoryError("SECRET-BODY"), ArithmeticError("SECRET-BODY"), KeyError("SECRET-BODY"), TypeError("SECRET-BODY"),
                                     ZeroDivisionError("SECRET-BODY"), AttributeError("SECRET-BODY"), RuntimeError("SECRET-BODY")], ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("path", sorted(RULES))
def test_whatever_a_validator_raises_about_a_request_body_is_a_422_that_does_not_quote_it(path, failure, reports, monkeypatch, caplog):
    rule = next(r for r in mutation_policy.MUTATIONS if re.fullmatch(r.pattern, path))

    def explode(body):
        raise failure
    monkeypatch.setattr(rule.model, "model_validate", explode)
    with caplog.at_level(logging.DEBUG, logger=mutation_policy.log.name):
        manager, refused = budget(path, {"whatever": 1})
    assert refused is not None and refused.status_code == 422
    error = refused.detail["error"]
    assert error["code"] == "invalid_request" and error["message"].startswith("Check the signal parameters") and error["details"] == []
    assert "SECRET-BODY" not in str(refused.detail)
    assert len(manager.limits) == RULES[path]                       # recipe rules charge first; the others charge only a body they accepted
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is None and "SECRET-BODY" not in warnings[0].getMessage()
    assert type(failure).__name__ in warnings[0].getMessage()


def test_an_unexpected_failure_is_reported_once_per_model_and_exception_type_not_once_per_request(reports, monkeypatch, caplog):
    path = "/signal-generator/nr-test-models/resolve"
    rule = next(r for r in mutation_policy.MUTATIONS if re.fullmatch(r.pattern, path))
    kinds = [OverflowError, OverflowError, RecursionError, OverflowError, RecursionError, KeyError]
    calls = iter(kinds)

    def explode(body):
        raise next(calls)("x")
    monkeypatch.setattr(rule.model, "model_validate", explode)
    with caplog.at_level(logging.DEBUG, logger=mutation_policy.log.name):
        for _ in kinds:
            _, refused = budget(path, {})
            assert refused.status_code == 422
    assert [r.getMessage().split(" validation raised ")[1].split(" ")[0] for r in caplog.records if r.levelno >= logging.WARNING] == ["OverflowError", "RecursionError", "KeyError"]
    assert all(r.exc_info is None for r in caplog.records)


def test_ordinary_validation_errors_are_still_explained_and_not_reported(reports, caplog):
    with caplog.at_level(logging.DEBUG, logger=mutation_policy.log.name):
        manager, refused = budget("/signal-generator/nr-test-models/resolve", recipe_with(boost=10 ** 400))
    assert refused.status_code == 422 and refused.detail["error"]["message"] == "request validation failed"
    assert [d["field"] for d in refused.detail["error"]["details"]] == ["nb_iot.boost_db"]
    assert manager.limits == [("nr-test-model:ip", 120)] and not caplog.records


@pytest.mark.parametrize("path,body", [("/signal-generator/nr-test-models/resolve", recipe_with(boost=10 ** 400)),
                                       ("/signal-generator/batches", {"configs": [{"nr_test_model": recipe_with(boost=10 ** 400)}]}),
                                       ("/signal-generator/signals", {"nr_test_model": recipe_with(boost=10 ** 400)}),
                                       ("/signal-generator/signals", {"nr_test_model": recipe_with(gap=10 ** 400)})])
def test_the_budget_judges_a_huge_number_as_a_422_without_any_report(path, body, reports, caplog):
    with caplog.at_level(logging.DEBUG, logger=mutation_policy.log.name):
        manager, refused = budget(path, body)
    assert refused is not None and refused.status_code == 422 and len(manager.limits) == 1 and not caplog.records

