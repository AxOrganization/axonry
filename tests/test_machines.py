"""Rented machines (pool/vast.py) and arbos account placement, against fakes of the vast API and of ssh."""
import json

import pytest

from axonry.pool.machine import MachineError

from axonry.agent.arbos import ArbosError, ArbosRunner
from axonry.agent.base import RunHandle
from axonry.orchestrator.db import DB
from axonry.pool.vast import Fleet, VastError


class FakeAPI:
    def __init__(self):
        self.created, self.destroyed, self.status, self.keys, self.taken = [], [], {}, [], None
        self.taken_msg = "no_such_ask"

    def ensure_ssh_key(self, pub):
        self.keys.append(pub)

    def offers(self, gpu_name, gpus, min_rel, disk, min_down, exclude=(), verified_only=False, min_cuda=0.0):
        return [{"id": 11, "machine_id": 1, "dph_total": 3.0}, {"id": 12, "machine_id": 2, "dph_total": 3.5}]

    def create(self, offer_id, image, disk, label):
        if offer_id == self.taken:
            raise VastError(f"PUT /asks/{offer_id}/: 400 {self.taken_msg}")
        self.created.append(offer_id)
        iid = f"i{len(self.created)}"
        self.status[iid] = {"actual_status": "loading"}
        return iid

    def show(self, iid):
        return self.status.get(iid)

    def destroy(self, iid):
        self.destroyed.append(iid)
        self.status.pop(iid, None)


class FakeRemote:
    def __init__(self):
        self.ready, self.failed, self.commands, self.drop, self.unreachable = False, False, [], 0, set()

    def ssh(self, m, key, command, timeout=0, attempts=3):
        if (m.ip, m.port) in self.unreachable:
            raise VastError("ssh: Connection closed by remote host")
        self.commands.append(command)
        if "READY" in command:
            return "FAILED\nsetup failed at line 9" if self.failed else ("1700000000" if self.ready else "")
        return ""

    def scp(self, m, key, src, dst, timeout=0):
        if self.drop:
            self.drop -= 1
            raise VastError("scp failed: lost connection")
        self.commands.append(f"scp {dst}")


def fleet(tmp_path, clock):
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA orchestrator")
    api, remote = FakeAPI(), FakeRemote()
    f = Fleet(api, DB(tmp_path / "db.sqlite"), image="img", disk_gb=100, ssh_key=str(tmp_path / "k"),
              bundle=lambda: tmp_path / "setup.tar", state_dir=tmp_path / "m", setup_minutes=45,
              clock=lambda: clock[0], remote=(remote.ssh, remote.scp), items=lambda: tmp_path / "items.tar")
    return f, api, remote


def test_a_machine_is_rented_set_up_and_released(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    assert api.created == [11] and api.keys == ["ssh-ed25519 AAAA orchestrator"]       # cheapest offer
    assert f.advance("qwen35-0.8b-think") == "booting"                                                 # instance still loading
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4 ",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    remote.drop = 1                                   # a dropped connection: retried, not a failure
    assert f.advance("qwen35-0.8b-think") == "booting" and api.destroyed == []
    assert f.advance("qwen35-0.8b-think") == "setting_up"
    assert any("setup.sh 8" in c for c in remote.commands) and "scp /root/axonry-setup.tar" in remote.commands
    assert not any("axonry-items" in c for c in remote.commands)          # a run's machine never gets the items
    assert f.advance("qwen35-0.8b-think") == "setting_up"
    remote.ready = True
    assert f.advance("qwen35-0.8b-think") == "ready"
    m = f.machine(f.get("qwen35-0.8b-think"))
    assert (m.ip, m.port, m.user) == ("198.51.100.4", 40022, "root")
    f.release("qwen35-0.8b-think")
    assert api.destroyed == ["i1"] and f.get("qwen35-0.8b-think") is None and f.active() == []


def test_a_machine_that_fails_setup_is_destroyed_and_never_rented_again(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    f.advance("qwen35-0.8b-think")
    remote.failed = True
    assert f.advance("qwen35-0.8b-think") == "failed" and api.destroyed == ["i1"] and f.get("qwen35-0.8b-think") is None
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)                       # vast machine 1 failed: the next offer
    assert api.created == [11, 12]
    clock[0] += 46 * 60                               # and one that never gets ready in time is dropped too
    assert f.advance("qwen35-0.8b-think") == "failed" and api.destroyed == ["i1", "i2"]


def test_a_machine_whose_container_cannot_start_is_dropped_at_once(tmp_path):
    f, api, _ = fleet(tmp_path, [1000.0])
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    assert f.advance("qwen35-0.8b-think") == "booting"               # still loading
    api.status["i1"] = {"actual_status": "created", "status_msg": "Error response from daemon: failed to inject CDI devices"}
    assert f.advance("qwen35-0.8b-think") == "failed" and api.destroyed == ["i1"]


def test_a_host_whose_direct_port_is_unreachable_is_used_through_the_vast_relay(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4", "ssh_host": "ssh9.vast.ai",
                        "ssh_port": 16808, "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    remote.unreachable.add(("198.51.100.4", 40022))
    assert f.advance("qwen35-0.8b-think") == "setting_up"
    m = f.machine(f.get("qwen35-0.8b-think"))
    assert (m.ip, m.port) == ("ssh9.vast.ai", 16808)


def test_a_host_that_never_installs_our_key_is_given_up(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    f.ssh = lambda *a, **k: (_ for _ in ()).throw(VastError("root@x: Permission denied (publickey)."))
    assert f.advance("qwen35-0.8b-think") == "booting"                # at first: sshd may not have the key yet
    clock[0] += 16 * 60
    assert f.advance("qwen35-0.8b-think") == "failed" and api.destroyed == ["i1"]


def test_a_vanished_machine_is_noticed(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    remote.ready = True
    f.advance("qwen35-0.8b-think")
    f.advance("qwen35-0.8b-think")
    assert not f.vanished("qwen35-0.8b-think")
    api.status["i1"] = {"actual_status": "exited"}
    assert f.vanished("qwen35-0.8b-think")


def test_an_offer_taken_meanwhile_is_skipped(tmp_path):
    f, api, _ = fleet(tmp_path, [0.0])
    api.taken = 11                                    # rented by someone else between the search and the order
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    assert api.created == [12] and f.get("qwen35-0.8b-think")["usd_hour"] == 3.5
    api.taken_msg = "GPU conflict: requesting already rented gpus {241322}"   # an offer still listed, its GPUs gone
    f.rent("s2", "RTX 4090", 8)
    assert api.created == [12, 12]


def test_no_offer_is_an_error(tmp_path):
    f, api, _ = fleet(tmp_path, [0.0])
    api.offers = lambda *a: []
    with pytest.raises(VastError):
        f.rent("qwen35-0.8b-think", "RTX 4090", 8)


def test_runs_go_to_the_arbos_account_with_the_most_room():
    calls = []

    def run(account, args, timeout):
        calls.append(account)
        if args[:3] == ["--json", "project", "list"]:
            return 0, "[]", ""
        if args[:3] == ["--json", "project", "create"]:
            return 0, json.dumps({"id": f"p-{account}"}), ""
        return 0, json.dumps({"id": "chat1"}), ""
    r = ArbosRunner({"a": 2, "b": 1}, runner=run)
    assert r.capacity == 3
    busy = []
    for want in ("a", "a", "b"):          # room: a 2, b 1 -> a; then a 1, b 1 -> a (first); then b
        account = r.pick(busy)
        assert account == want
        busy.append(RunHandle("s", "p", "c", {"account": account}))
    with pytest.raises(ArbosError):
        r.pick(busy)
    calls.clear()
    h = r.prepare("s9", "5Habc", busy[:2])
    assert h.data == {"account": "b"} and h.workspace == "p-b" and set(calls) == {"b"}


def test_a_champion_after_a_disqualified_one_gets_the_next_number(tmp_path):
    base = "qwen35-0.8b-think"
    db = DB(tmp_path / "db.sqlite")
    db.add_submission("0000000001-0001", base, "hk1", "ck", "PUBLISHED", {}, "d1", (1, 1))
    db.add_champion(base, "0000000001-0001", "hk1", 1, {})
    db.update("0000000001-0001", "DISQUALIFIED")
    db.add_champion(base, "0000000002-0001", "hk2", 2, {})          # used to collide with the disqualified one
    assert [c["seq"] for c in db.champions(base)] == [1, 2] and db.champion(base)["sid"] == "0000000002-0001"

def test_a_state_snapshot_holds_what_is_still_in_the_wal(tmp_path):
    db = DB(tmp_path / "s.sqlite")
    db.set("cursor", 123)                       # committed, not yet checkpointed into the main file
    (tmp_path / "copy.sqlite").write_bytes(db.snapshot())
    assert DB(tmp_path / "copy.sqlite").get("cursor") == 123


def test_sample_parses_gpus_status_line_and_done(monkeypatch):
    from axonry.pool import machine
    from axonry.pool.machine import MachineError

    def remote(gpus, rest):
        def ssh(m, key, command, **k):
            if command.startswith("timeout -k 2 55 nvidia-smi"):
                if isinstance(gpus, Exception):
                    raise gpus
                return gpus
            return rest
        return ssh
    status = "---\n\n# Status: " + " ".join(f"w{i}" for i in range(30)) + "\nmore\n"
    monkeypatch.setattr(machine, "ssh", remote("97, 20000, 24564\n0, 3, 24564\n", status))
    got = machine.sample(None, None)
    assert got["gpus"] == [{"util_pct": 97, "mem_used_mib": 20000, "mem_total_mib": 24564},
                           {"util_pct": 0, "mem_used_mib": 3, "mem_total_mib": 24564}]
    assert got["note"] == " ".join(f"w{i}" for i in range(20)) and got["done"] is False
    monkeypatch.setattr(machine, "ssh", remote("5, 1, 2\n", "done\n---\n"))
    got = machine.sample(None, None)
    assert got["note"] is None and got["done"] is True
    monkeypatch.setattr(machine, "ssh", remote("[N/A], 23314, 24564\n", "done\n---\n"))
    got = machine.sample(None, None)                     # a figure the driver cannot give hides nothing else
    assert got["gpus"] == [{"util_pct": None, "mem_used_mib": 23314, "mem_total_mib": 24564}] and got["done"] is True
    monkeypatch.setattr(machine, "ssh", remote(MachineError("`nvidia-smi` timed out"), "done\n---\nstill here\n"))
    got = machine.sample(None, None)                     # a sick GPU hangs nvidia-smi: the done flag still comes
    assert got["gpus"] is None and got["done"] is True and got["note"] == "still here"


def test_offers_leave_out_hosts_vast_deverified():
    from axonry.pool.vast import VastAPI
    api = VastAPI("k")
    api.call = lambda *a, **k: {"offers": [{"id": 1, "verification": "deverified"}, {"id": 2, "verification": "verified"},
                                           {"id": 3, "verification": "unverified"}]}
    assert [o["id"] for o in api.offers("RTX 4090", 4, 0.95, 150, 200)] == [2, 3]


def test_a_machine_downloads_the_bundle_itself_and_checks_it(tmp_path):
    import hashlib
    import subprocess
    import sys

    from axonry.pool.vast import FETCH
    clock = [1000.0]
    (tmp_path / "setup.tar").write_bytes(b"bundle")
    f, api, remote = fleet(tmp_path, clock)
    f.bundle_url = lambda path: "https://r2.example/setup.tgz?sig=1"
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    assert f.advance("qwen35-0.8b-think") == "setting_up"
    sha = hashlib.sha256(b"bundle").hexdigest()
    assert not any(c.startswith("scp ") for c in remote.commands)           # nothing sent from here
    assert any("r2.example" in c and sha in c and "setup.sh 8" in c for c in remote.commands)
    url, dest = (tmp_path / "setup.tar").as_uri(), str(tmp_path / "got.tgz")
    assert subprocess.run([sys.executable, "-c", FETCH, url, dest, sha]).returncode == 0
    assert subprocess.run([sys.executable, "-c", FETCH, url, dest, "0" * 64], capture_output=True).returncode != 0


def test_a_running_machine_whose_sshd_never_answers_is_given_up(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("qwen35-0.8b-think", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    remote.unreachable = {("198.51.100.4", 40022)}
    assert f.advance("qwen35-0.8b-think") == "booting"          # sshd may still be starting
    clock[0] += 9 * 60
    assert f.advance("qwen35-0.8b-think") == "booting"
    clock[0] += 2 * 60                                          # 11 min running, never reachable: a broken host
    assert f.advance("qwen35-0.8b-think") == "failed" and api.destroyed == ["i1"]


def test_an_unread_sample_keeps_the_last_reading():
    from types import SimpleNamespace

    from axonry.orchestrator.service import Orchestrator
    last = {"gpus": [{"util_pct": 97, "mem_used_mib": 20000, "mem_total_mib": 24564}], "note": "screening",
            "done": False, "at": 1.0}
    orch = SimpleNamespace(now=lambda: 5.0)
    got = Orchestrator._sample(orch, "s", {"gpus": None, "note": None, "done": False}, last)   # GPUs unread
    assert got == {"gpus": last["gpus"], "note": None, "done": False, "at": 5.0}
    assert Orchestrator._sample(orch, "s", RuntimeError("timed out"), last) == last   # unreachable: the last
    assert Orchestrator._sample(orch, "s", None, last) == last                          # reading, with its time


def test_machines_are_sampled_side_by_side(tmp_path, monkeypatch):
    import threading
    import time

    from axonry.pool import Machines
    from axonry.pool.machine import MachineError
    db = DB(tmp_path / "db.sqlite")
    for p in ("a", "b", "c"):
        db.add_machine(p, f"v-{p}", "m", "RTX 4090", 4, 1.0, 0)
        db.set_machine(p, state="ready", ssh_host="198.51.100.4", ssh_port=1)

    class Vast:
        def machine(self, row):
            return row["name"]
    seen, lock = [], threading.Lock()

    def slow(m, key):
        with lock:
            seen.append(m)
        time.sleep(0.3)
        if m == "b":
            raise MachineError("timed out")
        return {"gpus": [], "note": m, "done": False}
    monkeypatch.setattr("axonry.pool.sample", slow)
    t = time.time()
    got = Machines(db, "k", None, Vast()).sample_all(["a", "b", "c", "gone"])
    assert time.time() - t < 0.6 and sorted(seen) == ["a", "b", "c"]            # in parallel, not 0.9 s in a row
    assert got["a"]["note"] == "a" and isinstance(got["b"], MachineError) and "gone" not in got

def test_an_evaluation_machine_gets_the_benchmark_items(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("eval", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    assert f.advance("eval") == "setting_up"
    assert "scp /root/axonry-items.tar" in remote.commands
    assert any("setup.sh 8 /root/axonry-items.tar" in c for c in remote.commands)


def test_the_benchmark_sources_stay_out_of_the_setup_bundle():
    from pathlib import Path

    from axonry.pool.machine import NOT_BUNDLED
    src = Path(__file__).resolve().parents[1] / "src"
    assert all((src / rel).is_file() for rel in NOT_BUNDLED)         # a rename must not ship them silently


def test_an_evaluation_machine_is_rented_only_once_the_items_are_built(tmp_path):
    from axonry.pool import Machines
    db = DB(tmp_path / "db.sqlite")
    rented = []

    class Vast:
        def rent(self, purpose, gpu_name, gpus):
            rented.append(purpose)

    def broken():
        raise RuntimeError("benchmark items: the source is unreachable")
    with pytest.raises(RuntimeError):
        Machines(db, "k", None, Vast(), items=broken).rent("eval", "RTX 4090", 8)
    assert rented == []                                         # no machine taken (and none blamed) for our failure
    Machines(db, "k", None, Vast(), items=broken).rent("0000000101-0001", "RTX 4090", 4)
    assert rented == ["0000000101-0001"]                        # a run's machine never needs the items


def test_items_failing_their_check_release_the_machine_without_blaming_it(tmp_path):
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    f.rent("eval", "RTX 4090", 8)
    api.status["i1"] = {"actual_status": "running", "public_ipaddr": "198.51.100.4",
                        "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    assert f.advance("eval") == "setting_up"
    real = remote.ssh
    remote.ssh = lambda m, key, command, **k: "ITEMS\nbench:x: items missing" if "ITEMS_FAILED" in command \
        else real(m, key, command, **k)
    f.ssh = remote.ssh
    assert f.advance("eval") == "failed" and api.destroyed == ["i1"]
    gone = f.db.machines(states=("gone",))[-1]
    assert not gone["note"].startswith("failed")                # never rented again only for its own failures


def test_a_failed_check_is_never_taken_for_a_vanished_machine(tmp_path):
    from axonry.pool import Machines
    from axonry.pool.machine import MachineError
    db = DB(tmp_path / "db.sqlite")
    for p in ("a", "b"):
        db.add_machine(p, f"v-{p}", "m", "RTX 4090", 4, 1.0, 0)

    class Vast:
        def vanished(self, purpose, row):
            if purpose == "b":
                raise MachineError("vast.ai API unreachable")
            return row["provider_id"] == "v-a"
    got = Machines(db, "k", None, Vast()).vanished_all(["a", "b"])
    assert got["a"] is True and isinstance(got["b"], MachineError)   # the service acts only on True


def test_a_step_uses_the_status_read_beforehand():
    from axonry.agent import supervisor
    from axonry.agent.base import RunStatus

    class Runner:
        def poll(self, h):
            raise AssertionError("read again")
    task = {"run": {"deadline": 10_000, "stall_minutes": 20, "wrap_minutes": [30, 10], "max_agent_usd": None},
            "sid": "s"}
    st = RunStatus(True, 900.0)
    assert supervisor.step(1000, task, None, Runner(), {}, False, False, st) == "running"
    with pytest.raises(MachineError):                     # a status that could not be read: that run retries
        supervisor.step(1000, task, None, Runner(), {}, False, False, MachineError("arbos down"))


def test_no_machine_is_rented_while_the_vast_credit_is_low(tmp_path):
    from axonry.pool import Machines, NoMachine
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    api.call = lambda method, path, body=None, query=None: {"credit": 5.0}
    pool = Machines(f.db, str(tmp_path / "k"), None, f)
    with pytest.raises(NoMachine, match="credit"):     # the queue waits; the entry is not backed off
        pool.rent("s1", "RTX 4090", 4)
    assert not api.created
    api.call = lambda method, path, body=None, query=None: {"credit": 100.0}
    clock[0] += 301
    pool.rent("s1", "RTX 4090", 4)
    assert api.created


def test_an_account_out_of_credit_makes_the_queue_wait(tmp_path):
    from axonry.pool import Machines, NoMachine
    clock = [1000.0]
    f, api, remote = fleet(tmp_path, clock)
    api.call = lambda method, path, body=None, query=None: {"credit": 100.0}
    api.taken, api.taken_msg = 11, b'{"error":"insufficient_credit"}'
    api.offers = lambda *a, **k: [{"id": 11, "machine_id": 1, "dph_total": 3.0}]
    with pytest.raises(NoMachine):
        Machines(f.db, str(tmp_path / "k"), None, f).rent("s1", "RTX 4090", 4)
