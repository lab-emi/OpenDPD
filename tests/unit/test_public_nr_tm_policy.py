"""Public Studio policy for the NR test-model routes: published literally, resolve is a stateless, rate-limited, validated POST."""
import asyncio
import re

import pytest

from opendpd.web import mutation_policy
from opendpd.web.policy import FILE_ID, ROUTES, SLUG, STATELESS_POSTS, allowed, expensive_request, reject

CATALOG, RESOLVE = "/signal-generator/nr-test-models", "/signal-generator/nr-test-models/resolve"


def test_catalog_and_resolve_are_published_by_literal_patterns_only():
    assert allowed("GET", CATALOG) and allowed("POST", RESOLVE)
    assert CATALOG in ROUTES["GET"] and RESOLVE in ROUTES["POST"]
    literal = [p for method in ("GET", "POST") for p in ROUTES[method] if "nr-test-models" in p]
    assert len(literal) == 2 and not any(placeholder in p for p in literal for placeholder in (SLUG, FILE_ID, "[", "("))


@pytest.mark.parametrize("method,path", [("POST", CATALOG), ("GET", RESOLVE), ("PUT", RESOLVE), ("PUT", CATALOG), ("DELETE", CATALOG),
                                         ("GET", CATALOG + "/"), ("GET", CATALOG + "/other"), ("POST", RESOLVE + "/x"), ("POST", RESOLVE + "s"),
                                         ("GET", CATALOG + "/../presets"), ("POST", "/signal-generator/nr-test-model/resolve")])
def test_nothing_else_under_the_prefix_is_published(method, path):
    assert not allowed(method, path)


def test_resolve_takes_no_shared_compute_slot_but_every_other_generator_post_does():
    assert STATELESS_POSTS == {RESOLVE}
    assert not expensive_request("POST", RESOLVE) and not expensive_request("GET", CATALOG)
    for path in ("/signal-generator/validate", "/signal-generator/batches", "/signal-generator/signals", "/signal-generator/import",
                 "/signal-generator/signals/sg-" + "a" * 64 + "/dataset"):
        assert expensive_request("POST", path), path
    assert expensive_request("GET", "/signal-generator/signals/sg-" + "a" * 64 + "/download")


def test_resolve_has_a_validating_rate_limited_rule_that_stores_nothing():
    rule = next(r for r in mutation_policy.MUTATIONS if re.fullmatch(r.pattern, RESOLVE))
    assert rule.model.__name__ == "NrTestModelConfig" and rule.estimate == "none" and rule.explain and rule.per_minute <= 120
    assert not any(re.fullmatch(r.pattern, CATALOG) for r in mutation_policy.MUTATIONS)


class Manager:
    def __init__(self):
        self.limits = []

    def rate_limit(self, key, limit=None):
        self.limits.append((key, limit))


class Tenant:
    ip_key = "ip"
    root = None


def budget(body):
    manager = Manager()
    asyncio.run(mutation_policy.enforce_mutation_budget(manager, Tenant(), RESOLVE, body))
    return manager.limits


def test_a_valid_recipe_is_rate_limited_per_ip_and_needs_no_storage_estimate():
    assert budget({"model": "NR-FR1-TM3.1a", "carriers": [{"scs_khz": 30, "bandwidth_mhz": 100}]}) == [("nr-test-model:ip", 120)]
    assert budget({}) == [("nr-test-model:ip", 120)]                                              # all defaults is a valid recipe


def test_an_invalid_recipe_is_a_422_with_the_readable_messages_before_any_rate_or_compute():
    from fastapi import HTTPException
    for body, text in [({"model": "NR-FR1-TM1.1", "carriers": [{"scs_khz": 60, "bandwidth_mhz": 5}]}, "5 MHz channel is not defined for 60 kHz"),
                       ({"model": "nope"}, "Input should be"), ({"carriers": []}, "at least 1 item"), ({"carriers": [{"scs_khz": 20}]}, "Input should be 15, 30 or 60"),
                       ({"unknown": 1}, "Extra inputs are not permitted")]:
        manager = Manager()
        with pytest.raises(HTTPException) as caught:
            asyncio.run(mutation_policy.enforce_mutation_budget(manager, Tenant(), RESOLVE, body))
        error = caught.value.detail["error"]
        assert caught.value.status_code == 422 and error["code"] == "invalid_request" and error["message"] == "request validation failed"
        assert text in " ".join(d["message"] for d in error["details"]) and manager.limits == []


def test_other_rules_keep_their_generic_message():
    from fastapi import HTTPException
    manager = Manager()
    with pytest.raises(HTTPException) as caught:
        asyncio.run(mutation_policy.enforce_mutation_budget(manager, Tenant(), "/signal-generator/signals", {"n_samples": 1}))
    assert caught.value.detail["error"]["message"].startswith("Check the signal parameters") and caught.value.detail["error"]["details"] == []


def test_reject_carries_details_only_when_given():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        reject(422, "code", "message")
    assert caught.value.detail["error"]["details"] == []
    with pytest.raises(HTTPException) as caught:
        reject(422, "code", "message", [{"field": "a", "message": "b"}])
    assert caught.value.detail["error"]["details"] == [{"field": "a", "message": "b"}]
