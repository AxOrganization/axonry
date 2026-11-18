import json

from axonry import pricing
from axonry.chain.extrinsics import Commitment, Deposit, parse_extrinsic
from axonry.common.digest import content_digest, content_digest_of_blobs
from axonry.common.ids import make_commitment, make_sid, parse_commitment
from axonry.intake import binding
from axonry.intake.github import submission_dir

HK = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
CK = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
TREASURY = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
DIGEST = "ab" * 32
COMP = json.load(open(__file__.rsplit("/", 2)[0] + "/WeightSpaceProgramming/rules.json"))
RULES = {"netuid": 7, "base": "qwen35-0.8b-think"}


def batch(*calls):
    return {"call_module": "Utility", "call_function": "batch_all", "call_args": [{"name": "calls", "value": list(calls)}]}


def remark(text, netuid=7):
    return commit_call(netuid, text)


def tao_pay(value, dest=TREASURY):
    return {"call_module": "Balances", "call_function": "transfer_keep_alive",
            "call_args": [{"name": "dest", "value": {"Id": dest}}, {"name": "value", "value": value}]}


def parse(call, success=True):
    return parse_extrinsic(call, signer=CK, block=100, extrinsic_index=3, timestamp=1_700_000_000, success=success,
                           treasury=TREASURY, netuid=7)


def test_digest_dir_equals_blobs(tmp_path):
    (tmp_path / "derive").mkdir()
    (tmp_path / "entry.json").write_text("{}")
    (tmp_path / "derive" / "a.py").write_text("x=1")
    (tmp_path / "derive" / "__pycache__").mkdir()
    (tmp_path / "derive" / "__pycache__" / "a.pyc").write_bytes(b"junk")
    assert content_digest(tmp_path) == content_digest_of_blobs({"entry.json": b"{}", "derive/a.py": b"x=1"})


def test_ids_roundtrip():
    assert make_sid(4812345, 7) == "0004812345-0007"
    assert parse_commitment(make_commitment(DIGEST, 583231)) == (DIGEST, 583231)
    assert make_commitment(DIGEST, 583231) == f"{DIGEST}:583231"
    for bad in ("bad:1", f"{DIGEST}:0", f"{DIGEST}:alice", f"axonry:{DIGEST}:1", f"{DIGEST}:1 "):
        assert parse_commitment(bad) is None


def test_parse_deposit_and_commitment():
    (d,) = parse(tao_pay(10**9))
    assert d == Deposit(100, 3, CK, "tao", 10**9, 1_700_000_000)
    (c,) = parse(remark(make_commitment(DIGEST, 42)))
    assert c == Commitment(100, 3, CK, DIGEST, 42, 1_700_000_000)
    assert parse(remark(make_commitment(DIGEST, 42), netuid=8)) == []          # another subnet's commitment
    both = parse(batch(tao_pay(5), remark(make_commitment(DIGEST, 42))))
    assert sorted(type(x).__name__ for x in both) == ["Commitment", "Deposit"]


def test_one_commitment_per_extrinsic_and_deposits_summed():
    """One chain position names one content version; two transfers in one batch both count."""
    got = parse(batch(remark(make_commitment(DIGEST, 1)), tao_pay(5), remark(make_commitment("cd" * 32, 1)),
                      tao_pay(7)))
    assert [x.content_digest for x in got if isinstance(x, Commitment)] == [DIGEST]
    assert [x.amount_rao for x in got if isinstance(x, Deposit)] == [12]


def test_parse_ignores_wrong_dest_or_failed():
    assert parse(tao_pay(1, dest=CK)) == []
    assert parse(tao_pay(1), success=False) == []
    assert parse(remark(make_commitment(DIGEST, 1)), success=False) == []


def test_parse_alpha_deposit():
    pay = {"call_module": "SubtensorModule", "call_function": "transfer_stake", "call_args": [
        {"name": "destination_coldkey", "value": TREASURY}, {"name": "hotkey", "value": HK},
        {"name": "origin_netuid", "value": 7}, {"name": "destination_netuid", "value": 7},
        {"name": "alpha_amount", "value": 5 * 10**9}]}
    (d,) = parse(pay)
    assert d.asset == "alpha" and d.amount_rao == 5 * 10**9 and d.coldkey == CK
    moved = {"call_module": "SubtensorModule", "call_function": "transfer_stake_and_hotkey", "call_args": [
        {"name": "destination_coldkey", "value": TREASURY}, {"name": "origin_hotkey", "value": HK},
        {"name": "destination_hotkey", "value": HK}, {"name": "origin_netuid", "value": "7"},
        {"name": "destination_netuid", "value": "7"}, {"name": "alpha_amount", "value": "3"}]}
    (d,) = parse(moved)                                        # as decoded on finney: numbers may come as strings
    assert d.asset == "alpha" and d.amount_rao == 3


def test_deposits_count_in_usd_and_a_run_by_gpu_hours_and_agent_use():
    assert pricing.deposit_musd(10**8, "tao", 500.0) == 50 * pricing.MUSD               # 0.1 TAO at 500 USD
    assert pricing.deposit_musd(10**10, "alpha", 500.0, alpha_tao=0.01) == 50 * pricing.MUSD
    rules = {"fee": {"usd_per_gpu_hour": 2, "slippage": 0.02}, "run": {"gpus": 4, "hours": 6, "max_agent_usd": 50}}
    assert pricing.hour_musd(rules) == 8 * pricing.MUSD
    assert pricing.run_cost_musd(rules) == 98_000_000                  # 48 GPU + 50 agent USD: what is held
    assert pricing.used_musd(rules, 5400) == 12 * pricing.MUSD                           # 1.5 h of GPUs
    assert pricing.used_musd(rules, 10 * 3600) == 48 * pricing.MUSD                      # never more than 6 h
    assert pricing.agent_musd(rules, 12.5) == 12_500_000 and pricing.agent_musd(rules, 80) == 50 * pricing.MUSD


def test_tao_usd_median_and_spread():
    fetch = {"a": lambda ts: 500.0, "b": lambda ts: 505.0, "c": lambda ts: (_ for _ in ()).throw(RuntimeError())}
    med, got = pricing.tao_usd_at(0, sources=("a", "b", "c"), fetch=fetch)
    assert med == 502.5 and set(got) == {"a", "b"}
    fetch["b"] = lambda ts: 600.0
    try:
        pricing.tao_usd_at(0, sources=("a", "b"), fetch=fetch)
        raise AssertionError("expected PriceError")
    except pricing.PriceError:
        pass


def test_submission_dir():
    assert submission_dir([f"submissions/{HK}/e1/entry.json", f"submissions/{HK}/e1/METHOD.md"])[:3] == (
        HK, "e1", f"submissions/{HK}/e1/")
    assert submission_dir(["README.md"])[3]
    assert submission_dir([f"submissions/{HK}/E1/x"])[3]          # upper case entry name is outside the pattern
    assert submission_dir([f"submissions/{HK}/e1/a", f"submissions/{HK}/e2/a"])[3]


def test_check_content():
    entry = {"schema": "axonry.entry/1", "base": "qwen35-0.8b-think"}
    ok = {"entry.json": json.dumps(entry).encode(), "METHOD.md": b"m"}
    assert binding.check_content(ok, RULES, COMP) == entry
    assert binding.check_content({"METHOD.md": b"m"}, RULES, COMP).code == binding.BAD_PR
    other = {**ok, "entry.json": json.dumps({**entry, "base": "other"}).encode()}
    assert binding.check_content(other, RULES, COMP).code == binding.BASE_NOT_ALLOWED


def test_a_deposit_is_priced_once_the_sources_are_back():
    """A failed price lookup for a deposit's moment is retried on the wall clock, not cached for good."""
    from axonry import pricing
    from axonry.orchestrator.cli import cached_price
    now, up, calls = [1000.0], [False], []

    def lookup(ts):
        calls.append(ts)
        if not up[0]:
            raise pricing.PriceError("need 2 price sources, got []")
        return 200.0, {"a": 200.0, "b": 200.0}
    price = cached_price(lookup=lookup, clock=lambda: now[0])
    for _ in range(2):
        try:
            price(1_800_000_000)
            raise AssertionError("expected no price")
        except pricing.PriceError:
            pass
    assert len(calls) == 1                         # not hammered within the retry window
    up[0], now[0] = True, now[0] + 61
    assert price(1_800_000_000)[0] == 200.0 and price(1_800_000_100)[0] == 200.0 and len(calls) == 2


def commit_call(netuid, text):
    """A Commitments.set_commitment call as the chain decodes it."""
    raw = text.encode()
    return {"call_module": "Commitments", "call_function": "set_commitment", "call_args": [
        {"name": "netuid", "value": netuid}, {"name": "info", "value": {"fields": [{f"Raw{len(raw)}": "0x" + raw.hex()}]}}]}
