import json
import types

import pytest
from bittensor_wallet import Keypair

from axonry.common import layout
from axonry.common.store import ImmutableObjectError, LocalStore
from axonry.publish import publisher
from axonry_validator import main as v

OWNER = Keypair.create_from_uri("//Alice")
HK = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def test_store_write_once(tmp_path):
    s = LocalStore(tmp_path, public=True)
    s.put("submissions/x/score.json", b"1")
    s.put("submissions/x/score.json", b"1")  # idempotent
    with pytest.raises(ImmutableObjectError):
        s.put("submissions/x/score.json", b"2")
    s.put(layout.WEIGHTS, b"a")
    s.put(layout.WEIGHTS, b"b")  # mutable


def test_weights_sign_verify_and_monotonic(tmp_path):
    s = LocalStore(tmp_path, public=True)
    doc = publisher.build_weights(7, "b", {"sid": "0000000001-0001", "hotkey": HK, "uid": 3}, 1, 100)
    signed = publisher.publish_weights(s, doc, OWNER)
    assert v.check(signed, OWNER.ss58_address, 7, -1) is None
    assert v.check(signed, OWNER.ss58_address, 8, -1)
    assert v.check({**signed, "sequence": 2}, OWNER.ss58_address, 7, -1) == "bad signature"
    assert v.check(signed, OWNER.ss58_address, 7, 5)
    with pytest.raises(publisher.PublishError):
        publisher.publish_weights(s, doc, OWNER)


def test_resolve_falls_back_to_uid0():
    doc = publisher.build_weights(7, "b", {"sid": "x", "hotkey": HK, "uid": 3}, 1, 100)
    assert v.resolve(doc, {3: HK}) == {3: 1.0}
    assert v.resolve(doc, {3: "someone else"}) == {0: 1.0}
    assert v.resolve(publisher.build_weights(7, "b", None, 1, 100), {}) == {0: 1.0}


def test_validator_cycle_keeps_last_valid(tmp_path):
    s = LocalStore(tmp_path / "pub", public=True)
    signed = publisher.publish_weights(s, publisher.build_weights(7, "b", {"sid": "x", "hotkey": HK, "uid": 3}, 1, 9),
                                       OWNER)
    a = types.SimpleNamespace(url="u", owner=OWNER.ss58_address, netuid=7, max_age=10**9, version_key=0)
    state = v.State(tmp_path / "state.json")
    calls = []
    sub = types.SimpleNamespace(query_map=lambda item, params: [(3, HK)])
    w = v.cycle(a, state, sub, None, fetcher=lambda u: signed, setter=lambda *x: calls.append(x[3]))
    assert w == {3: 1.0}
    tampered = {**signed, "weights": [{"uid": 9, "hotkey": None, "weight": 1.0}]}
    w = v.cycle(a, state, sub, None, fetcher=lambda u: tampered, setter=lambda *x: calls.append(x[3]))
    assert w == {3: 1.0} and calls == [{3: 1.0}, {3: 1.0}]


def test_publish_registry(tmp_path):
    from axonry import registry
    s = LocalStore(tmp_path, public=True)
    publisher.publish_registry(s)
    rules = registry.latest_rules()
    publisher.publish_index(s, rules)
    idx = json.loads(s.get(layout.INDEX))
    assert json.loads(s.get(idx["rules"])) == rules.doc          # index -> the rules version in force, no copy
    assert s.get(layout.base_def("qwen35-0.8b-think")) and s.get(layout.benchmark_def("math500", 1))


def test_the_owner_burns_a_share_beside_the_champion_and_validators_need_no_change():
    champ = {"sid": "s", "hotkey": "hk5", "uid": 5}
    doc = publisher.build_weights(99, "b", champ, 3, 100, burn=0.3)
    assert v.check(doc, None, 99, 2) is None or "signature" in v.check(doc, None, 99, 2)
    assert v.resolve(doc, {5: "hk5"}) == {5: 0.7, 0: 0.3}
    assert v.resolve(doc, {5: "other"}) == {0: 1.0}                     # deregistered champion: all burned
    assert v.resolve(publisher.build_weights(99, "b", champ, 3, 100, burn=1.0), {5: "hk5"}) == {0: 1.0}
    assert v.resolve(publisher.build_weights(99, "b", champ, 3, 100), {5: "hk5"}) == {5: 1.0}


def test_weights_are_set_once_per_half_epoch_from_block_zero():
    from axonry_validator.main import window
    assert window(0, 360) == window(179, 360) == 0 and window(180, 360) == 1      # windows of 180 blocks
    assert window(9546783, 360) == 9546783 // 180
    starts = [b for b in range(1, 2000) if window(b, 360) != window(b - 1, 360)]
    assert all(b % 180 == 0 for b in starts) and min(y - x for x, y in zip(starts, starts[1:])) > 100   # rate limit
