"""Owned GPU hosts (pool/owned.py) and the machine router (pool/__init__.py), against a fake host; the agent's ssh files
through a jump host; redaction of a run's secrets from its published log."""
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pytest

from axonry.agent import brief
from axonry.orchestrator.db import DB
from axonry.pool import Machines, NoMachine
from axonry.pool.machine import EVAL, Jump, Machine, MachineError
from axonry.pool.owned import OwnedPool
from axonry.publish import redact


class FakeHost:
    """host.sh as seen over ssh: detached steps finish when told to; containers come and go."""

    def __init__(self, gpus=8, name="NVIDIA GeForce RTX 4090"):
        self.gpus, self.name, self.done, self.fail, self.containers = gpus, name, set(), {}, {}
        self.started, self.jumps, self.copied, self.down, self.held = [], {}, [], False, {}

    def ssh(self, m, key, command, timeout=0, attempts=3):
        if self.down:
            raise MachineError("ssh: connect to host: No route to host")
        if m.jump:                                         # inside a container (through the jump host)
            assert m.jump and m.jump.user == "root" and m.ip == "127.0.0.1"
            return ""
        if command.startswith("sha256sum "):
            return self.held.get(command.split()[1], "") + "\n"
        if command.startswith("nvidia-smi"):
            return "\n".join([self.name] * self.gpus) + "\n"
        if not command.startswith("/srv/axonry/host/host.sh "):
            return ""
        args = command.split("/host.sh ", 1)[1].split()
        if args[0] == "start":
            step = args[1]
            if step in self.fail:
                return "failed\n" + self.fail.pop(step)
            if step in self.done:
                return "ok"
            self.started.append(step)
            return "running"
        if args[0] == "state":
            return self.containers.get(args[1], "missing")
        if args[0] == "run":
            self.containers[f"axonry-run{args[1]}"] = f"true {args[4]}"
            self.last_run = args
            return ""
        if args[0] == "jump":
            self.jumps[args[1]] = command.split(args[1], 1)[1].strip()
            return ""
        if args[0] == "stop":
            self.containers.pop(args[1], None)
            return ""
        raise AssertionError(command)

    def scp(self, m, key, srcs, dst, timeout=0):
        self.copied.append(dst)
        if len(srcs) == 1:
            self.held[dst] = hashlib.sha256(srcs[0].read_bytes()).hexdigest()


def pool(tmp_path, hosts, fake, clock):
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA orchestrator")
    (tmp_path / "setup.tgz").write_bytes(b"bundle")
    db = DB(tmp_path / "db.sqlite")
    (tmp_path / "items.tar").write_bytes(b"items")
    p = OwnedPool(hosts, db, image="img@sha256:1", ssh_key=str(tmp_path / "k"), bundle=lambda: tmp_path / "setup.tgz",
                  items=lambda: tmp_path / "items.tar", state_dir=tmp_path / "m", clock=lambda: clock[0], remote=(fake.ssh, fake.scp))
    return p, db


def finish(fake, prefix):
    fake.done |= {s for s in fake.started if s.startswith(prefix)}


def test_runs_get_fresh_containers_in_free_slots(tmp_path):
    clock, fake = [1000.0], FakeHost()
    p, db = pool(tmp_path, [{"name": "rig", "address": "203.0.113.5", "port": 9000, "role": "run", "china": True}],
                 fake, clock)
    assert p.rent("s1", "RTX 4090", 4) and p.rent("s2", "RTX 4090", 4)
    assert not p.rent("s3", "RTX 4090", 4)                                  # two slots of 4 GPUs: both taken
    assert not p.rent("s4", "RTX 5090", 4)                                  # another GPU model
    assert p.advance("s1") == "booting"                                     # the host is being initialized
    assert fake.started[0].startswith("init-") and "/srv/axonry/host/host.sh" in fake.copied
    assert "/srv/axonry/host/items.tar" not in fake.copied                  # a run host never gets benchmark items
    finish(fake, "init-")
    assert p.advance("s1") == "booting" and fake.started[-1].startswith("bake-")   # then the run image is baked
    finish(fake, "bake-")
    assert p.advance("s1") == "ready"
    assert fake.last_run[1:3] == ["1", "0,1,2,3"] and fake.last_run[3].startswith("axonry-run:")
    assert p.advance("s2") == "ready" and fake.last_run[1:3] == ["2", "4,5,6,7"]   # host ready: seconds
    m = p.hand_over("s2", "ssh-ed25519 BBBB agent", 2000)
    assert (m.ip, m.port, m.user) == ("127.0.0.1", 2302, "agent")
    assert (m.jump.host, m.jump.port, m.jump.user) == ("203.0.113.5", 9000, "axjump") and m.jump.proxy_command is None
    assert fake.jumps == {"2302": "'ssh-ed25519 BBBB agent'"}               # forwarded to its own port only
    assert not p.vanished("s2")
    fake.containers.pop("axonry-run2")                                      # host rebooted, container gone
    assert p.vanished("s2")
    p.release("s1")
    assert "axonry-run1" not in fake.containers and p.rent("s3", "RTX 4090", 4)   # slot 1 is free again
    assert db.machines()[-1]["provider_id"].startswith("owned:rig:1:")


def test_the_evaluation_host_keeps_its_container(tmp_path):
    clock, fake = [1000.0], FakeHost()
    p, db = pool(tmp_path, [{"name": "ev", "address": "203.0.113.6", "role": "eval"}], fake, clock)
    assert not p.rent(EVAL, "RTX 4090", 4)                                  # needs all 8 GPUs of the host
    assert not p.rent(EVAL, "RTX 5090", 8)                                  # and the rules' GPU model
    assert p.rent(EVAL, "RTX 4090", 8)
    assert p.advance(EVAL) == "booting"
    finish(fake, "init-")
    assert p.advance(EVAL) == "booting" and fake.started[-1].startswith("eval-")
    assert "/srv/axonry/host/items.tar" in fake.copied                      # the benchmark items: evaluation host only
    sha = fake.started[-1].split("-", 1)[1]
    finish(fake, "eval-")
    assert p.advance(EVAL) == "booting"                                     # marker there, container missing: again
    fake.containers["axonry-eval"] = f"true {sha}"
    assert p.advance(EVAL) == "ready"
    m = p.machine(p.get(EVAL))
    assert (m.ip, m.port, m.jump.user, m.jump.host) == ("127.0.0.1", 2222, "root", "203.0.113.6")
    p.release(EVAL, "idle")
    assert "axonry-eval" in fake.containers                                 # kept for the next evaluation
    assert p.rent(EVAL, "RTX 4090", 8) and p.advance(EVAL) == "ready"


def test_a_failed_host_step_fails_the_machine_and_is_tried_again(tmp_path):
    clock, fake = [1000.0], FakeHost()
    p, db = pool(tmp_path, [{"name": "rig", "address": "203.0.113.5", "role": "run"}], fake, clock)
    p.rent("s1", "RTX 4090", 4)
    assert p.advance("s1") == "booting"
    fake.fail[fake.started[-1]] = "GPUs in use by other processes: 4242 rgminer"
    assert p.advance("s1") == "failed" and p.get("s1") is None
    note = db.machines(states=("gone",))[-1]["note"]
    assert "rgminer" in note
    assert p.rent("s1", "RTX 4090", 4) and p.advance("s1") == "booting"    # an owned host is never given up
    fake.down = True
    assert p.advance("s1") == "booting" and not p.vanished("s1")           # unreachable: retried, not failed


def test_the_router_prefers_owned_hosts_and_waits_without_vast(tmp_path):
    clock, fake = [1000.0], FakeHost(gpus=4)
    p, db = pool(tmp_path, [{"name": "rig", "address": "203.0.113.5", "role": "run"}], fake, clock)

    class Vast:
        rented = []

        def rent(self, purpose, gpu_name, gpus):
            self.rented.append(purpose)
            db.add_machine(purpose, f"v{len(self.rented)}", "9", gpu_name, gpus, 2.0, clock[0])

        def release(self, purpose, note="released"):
            self.released = purpose
    machines = Machines(db, "k", p, Vast())
    machines.rent("s1", "RTX 4090", 4)
    machines.rent("s2", "RTX 4090", 4)
    assert Vast.rented == ["s2"]                                            # owned first, then vast.ai
    machines.release("s2")
    assert machines.vast.released == "s2"
    with pytest.raises(NoMachine):
        Machines(db, "k", p, None).rent("s3", "RTX 4090", 4)


def test_ssh_through_a_jump_host():
    j = Jump("203.0.113.5", 22, "root", "/kh", "nc -X 5 -x 127.0.0.1:1084 %h %p")
    m = Machine("s1", "127.0.0.1", 2301, "/kh1", jump=j)
    proxy = dict(a.split("=", 1) for a in m.ssh_args("/key")[1::2] if "=" in a)["ProxyCommand"]
    assert proxy.endswith("-W %h:%p root@203.0.113.5")
    assert "nc -X 5 -x 127.0.0.1:1084 203.0.113.5 22" in proxy             # the jump host's own proxy, filled in
    task = {"sid": "s1", "sandbox": {"host": "127.0.0.1", "port": 2301, "user": "agent", "host_key": "ssh-ed25519 C",
                                     "private_key": "KEY", "jump": {"host": "203.0.113.5", "port": 22,
                                                                     "user": "axjump", "host_key": "ssh-ed25519 H"}}}
    files = brief.ssh_files(task)
    cfg = files["config"].decode()
    assert "Host axonry-jump\n  HostName 203.0.113.5\n  Port 22\n  User axjump" in cfg and "ProxyJump axonry-jump" in cfg
    assert files["known_hosts"].decode() == "sandbox ssh-ed25519 C\naxonry-jump ssh-ed25519 H\n"
    del task["sandbox"]["jump"]
    assert "ProxyJump" not in brief.ssh_files(task)["config"].decode()


def test_a_published_agent_log_carries_none_of_the_runs_secrets(tmp_path):
    key = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n" \
          "QyNTUxOQAAACBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==\n-----END OPENSSH PRIVATE KEY-----\n"
    task = {"sid": "s1", "sandbox": {"host": "127.0.0.1", "port": 2301, "jump": {"host": "203.0.113.5"}},
            "r2": {"endpoint": "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
                   "access_key_id": "AKIDAKIDAKID1234", "secret_access_key": "s" * 64, "session_token": "t" * 300}}
    bundle = tmp_path / "axonry-task-s1.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for name, data in (("task.json", json.dumps(task).encode()), ("ssh/id_sandbox", key.encode())):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    log = tmp_path / "agent"
    log.mkdir()
    (log / "chat.json").write_text(json.dumps({"messages": [
        {"text": "cat ssh/config\nHost axonry-jump\n  HostName 203.0.113.5"},
        {"text": "cat task.json: " + json.dumps(task)},
        {"text": "cat id_sandbox\n" + key.splitlines()[1] + "\n" + key},
        {"text": "model digest " + "a" * 64 + " on 127.0.0.1:2301"}]}))
    redact.redact_dir(log, redact.secrets_in_bundle(bundle))
    out = (log / "chat.json").read_text()
    doc = json.loads(out)                                                   # still valid JSON
    for secret in ("203.0.113.5", "0123456789abcdef0123456789abcdef", "AKIDAKIDAKID1234", "s" * 64, "t" * 300,
                   "PRIVATE KEY", key.splitlines()[1]):
        assert secret not in out
    assert "a" * 64 in doc["messages"][3]["text"]                           # not a secret: kept
    assert re.search(r"\[redacted\]", out)


def test_benchmark_items_are_built_checked_and_packed(tmp_path, monkeypatch):
    """The orchestrator's pack for an evaluation machine: the toolkit's files of the items built from their sources,
    only when every benchmark matches its frozen manifest (a stand-in build here: no network)."""
    import tarfile as tar_mod

    from axonry.evaluation import benchdata

    def built(competition, spec, work):
        d = Path(work) / "data"
        d.mkdir(parents=True, exist_ok=True)
        for b in spec:
            (d / f"bench_{b['task'].split(':')[1]}.jsonl").write_text("{}\n")
        return {b["key"]: "m-" + b["task"] for b in spec}
    monkeypatch.setattr(benchdata, "_built", built)
    docs = [{"id": "amc23", "version": 1, "task": "bench:amc23", "items_manifest_sha256": "m-bench:amc23"}]
    out = benchdata.build(tmp_path / "comp", docs, tmp_path / "state")
    with tar_mod.open(out) as tf:
        assert tf.getnames() == ["data", "data/bench_amc23.jsonl"]
    assert benchdata.build(tmp_path / "comp", docs, tmp_path / "state") == out    # built once
    bad = [{**docs[0], "items_manifest_sha256": "frozen-other"}]
    with pytest.raises(RuntimeError, match="amc23@1: items m-bench:amc2 != frozen frozen-other"):
        benchdata.build(tmp_path / "comp", bad, tmp_path / "state")


def test_items_are_written_where_the_toolkit_reads_them(tmp_path, monkeypatch):
    """write_items: each benchmark's rows from benchmarks/sources.py into the toolkit's data dir (stand-in toolkit)."""
    import sys
    import types

    from axonry.evaluation import benchdata
    written = {}
    wsp = types.ModuleType("wsp")
    data = types.SimpleNamespace(DATA=tmp_path, manifest=lambda task: {"sha256": "m-" + task})
    common = types.SimpleNamespace(write_jsonl=lambda path, rows: written.setdefault(path, rows))
    sources = types.SimpleNamespace(rows=lambda name: [{"id": name + "_0"}])
    monkeypatch.setitem(sys.modules, "wsp", wsp)
    monkeypatch.setitem(sys.modules, "wsp.data", data)
    monkeypatch.setitem(sys.modules, "wsp.common", common)
    monkeypatch.setitem(sys.modules, "axonry.benchmarks.sources", sources)
    wsp.data, wsp.common = data, common
    import axonry.benchmarks as pkg
    monkeypatch.setattr(pkg, "sources", sources, raising=False)
    got = benchdata.write_items([{"key": "amc23@1", "task": "bench:amc23"}])
    assert got == {"amc23@1": "m-bench:amc23"} and written == {tmp_path / "bench_amc23.jsonl": [{"id": "amc23_0"}]}


def test_a_file_the_host_already_holds_is_not_sent_again(tmp_path):
    clock, fake = [1000.0], FakeHost()
    host = [{"name": "rig", "address": "203.0.113.5", "port": 9000, "role": "run", "china": True}]
    p, _ = pool(tmp_path, host, fake, clock)
    assert p.rent("s1", "RTX 4090", 4) and p.advance("s1") == "booting"
    bundles = [d for d in fake.copied if "/setup-" in d]
    assert len(bundles) == 1
    p2, _ = pool(tmp_path, host, fake, clock)                     # a restarted orchestrator, same bundle
    assert p2.advance("s1") == "booting"
    assert [d for d in fake.copied if "/setup-" in d] == bundles  # the host holds it: no second upload


def test_a_pack_in_an_older_layout_is_built_again(tmp_path, monkeypatch):
    import tarfile as tar_mod

    from axonry.evaluation import benchdata
    docs = [{"id": "gsm8k500", "version": 1, "task": "bench:gsm8k500", "items_manifest_sha256": "m-bench:gsm8k500"}]
    s = benchdata.spec(docs)
    old = tmp_path / "state" / f"benchdata-{__import__('hashlib').sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]}.tar"
    old.parent.mkdir(parents=True)
    src = tmp_path / "old" / "data"
    src.mkdir(parents=True)
    (src / "bench_gsm8k.jsonl").write_text("{}\n")                 # how the toolkit cached it before
    with tar_mod.open(old, "w") as tf:
        tf.add(src, arcname="data")

    def built(competition, spec, work):
        d = Path(work) / "data"
        d.mkdir(parents=True, exist_ok=True)
        for b in spec:
            (d / f"bench_{b['task'].split(':')[1]}.jsonl").write_text("{}\n")
        return {b["key"]: "m-" + b["task"] for b in spec}
    monkeypatch.setattr(benchdata, "_built", built)
    out = benchdata.build(tmp_path / "comp", docs, tmp_path / "state")
    with tar_mod.open(out) as tf:
        assert sorted(tf.getnames()) == ["data", "data/bench_gsm8k500.jsonl"]
