"""Whole flow against fakes: deposit + commitment + PR -> run charged by the hour -> review -> sandbox + temp creds -> agent run -> upload -> admission ->
benchmarks -> decision -> public files + signed weights, and the miner CLI reading them back."""
import json
import random
from pathlib import Path

import pytest
from axonry import pricing, registry

from bittensor_wallet import Keypair

from axonry.chain.extrinsics import Commitment, Deposit
from axonry.common import layout
from axonry.common.digest import content_digest_of_blobs
from axonry.common.store import LocalStore
from axonry.creds import TempCreds
from axonry.common.canon import sha256_hex
from axonry.intake.github import PRCandidate
from axonry.miner import cli as miner
from axonry.orchestrator.db import DB
from axonry.orchestrator.service import Deps, Orchestrator
from axonry.pool.machine import Machine
from axonry_validator import main as validator

OWNER = Keypair.create_from_uri("//Alice")
OWNER_COLDKEY = Keypair.create_from_uri("//AliceStash").ss58_address
TREASURY = Keypair.create_from_uri("//Treasury").ss58_address
NETUID = 99
HOTKEYS = {n: Keypair.create_from_uri(f"//hk{n}").ss58_address for n in "ABCD"}
COLDKEYS = {n: Keypair.create_from_uri(f"//ck{n}").ss58_address for n in "ABCD"}
UIDS = {HOTKEYS[n]: i + 1 for i, n in enumerate("ABCD")}
COMP_RULES = json.load(open(__file__.rsplit("/", 2)[0] + "/WeightSpaceProgramming/rules.json"))

# what each miner's agent will produce: quality = share of extra items answered right; spec = edit.json content
PLANS = {"A": {"quality": 0.25, "spec": {"base": "qwen35-0.8b-think", "ops": ["a"]}},
         "B": {"quality": 0.9, "spec": {"base": "qwen35-0.8b-think", "ops": ["too_big"]}},
         "C": {"quality": 0.25, "spec": {"base": "qwen35-0.8b-think", "ops": ["a"]}},   # same weights as A
         "D": {"quality": 0.27, "spec": {"base": "qwen35-0.8b-think", "ops": ["d"]}},  # not better than A
         "E": {"quality": 0.5, "spec": {"base": "qwen35-0.8b-think", "ops": ["e"]}}}   # stacks on A, beats it
HOTKEYS["E"] = Keypair.create_from_uri("//hkE").ss58_address
COLDKEYS["E"] = Keypair.create_from_uri("//ckE").ss58_address
UIDS[HOTKEYS["E"]] = 5
HOTKEYS["F"] = Keypair.create_from_uri("//hkF").ss58_address          # registered and funded, never commits
COLDKEYS["F"] = Keypair.create_from_uri("//ckF").ss58_address
UIDS[HOTKEYS["F"]] = 6



def make_manifest(out_dir):
    """The upload manifest exactly as the competition toolkit's wsp.upload writes it."""
    files = [{"path": "edit/" + p.relative_to(out_dir).as_posix(), "size": p.stat().st_size,
              "sha256": sha256_hex(p.read_bytes())} for p in sorted(out_dir.rglob("*")) if p.is_file()]
    return {"schema": COMP_RULES["upload"]["schema"], "files": files}


class Clock:
    t = 1_800_000_000

    def __call__(self):
        return self.t


class FakeChain:
    def __init__(self):
        self.head, self.blocks = 100, {}

    def finalized_head(self):
        return self.head

    def items_in_block(self, b, treasury, netuid):
        return self.blocks.get(b, [])

    def owner_of(self, hk, block):
        return next((COLDKEYS[n] for n, h in HOTKEYS.items() if h == hk), OWNER_COLDKEY)

    def uid_of(self, hk, netuid, block=None):
        return UIDS.get(hk)

    def registration_owner(self, hk, netuid, block):
        return 1, self.owner_of(hk, 1)

    def uid_hotkeys(self, netuid, block=None):
        return {u: h for h, u in UIDS.items()}

    def alpha_price(self, netuid, block):
        return 0.01


def blobs_for(n, parent=None):
    return {"entry.json": json.dumps({"schema": "axonry.entry/1", "base": "qwen35-0.8b-think",
                                      "parent": parent}).encode(),
            "METHOD.md": f"method of {n}".encode()}


USER_IDS = {"u": 1, "mallory": 666}       # GitHub login -> numeric id


class FakeGitHub:
    """PRs with a moving head; commits map to submission directory contents."""
    repo = "owner/WeightSpaceProgramming"

    def __init__(self):
        self.commits, self.prs, self.tags, self.merged, self.comments = {}, {}, {}, [], []
        self.approve_all, self.verdicts = True, {}      # (PR, content digest) -> "approved" | "changes"

    def add_pr(self, number, hotkey, commit, blobs):
        self.commits[commit] = blobs
        self.prs[number] = {"number": number, "state": "open", "head_sha": commit, "mergeable": True,
                            "author": "u", "merged": False, "dir": f"submissions/{hotkey}/e1/"}

    def push(self, number, commit, blobs):
        self.commits[commit] = blobs
        self.prs[number]["head_sha"] = commit

    def pr(self, number):
        p = self.prs[number]
        return {**{k: v for k, v in p.items() if k != "dir"}, "author_id": p.get("author_id", USER_IDS[p["author"]])}

    def pr_files(self, number):
        d = self.prs[number]["dir"]
        return [d + rel for rel in self.commits[self.prs[number]["head_sha"]]]

    def dir_blobs(self, commit, directory):
        return dict(self.commits[commit])

    def review_state(self, number, digest, directory, reviewers):
        if (number, digest) in self.verdicts:          # reviews live on a PR, like GitHub's
            return self.verdicts[(number, digest)], "reviewer note"
        return ("approved", None) if self.approve_all else (None, None)

    def lock_tag(self, sid, commit):
        assert self.tags.get(sid, commit) == commit
        self.tags[sid] = commit

    def merge(self, commit, title):
        assert commit in self.tags.values(), "only a locked commit is merged"
        self.merged.append(int(title.rsplit("#", 1)[1].rstrip(")")))

    def comment(self, number, text):
        self.comments.append((number, text))

    def close(self, number):
        self.prs[number]["state"] = "closed"

    def candidates(self):
        """What intake.github.scan reports: open PRs and the content digest at their head."""
        out = []
        for n, p in self.prs.items():
            if p["state"] == "open":
                hk = p["dir"].split("/")[1]
                out.append(PRCandidate(n, p["author"], hk, "e1", p["dir"], None, p["head_sha"],
                                       content_digest_of_blobs(self.commits[p["head_sha"]]), p.get("title", ""),
                                       self.pr(n)["author_id"], p.get("opened_at")))
        return out


class FakeFleet:
    """Rented machines, ready at once; `lose` makes vast drop one."""
    def __init__(self, db):
        self.db, self.lost, self.released, self.done = db, set(), [], set()

    def get(self, purpose):
        return next((m for m in self.db.machines() if m["name"] == purpose), None)

    def active(self):
        return self.db.machines()

    def rent(self, purpose, gpu_name, gpus):
        self.db.add_machine(purpose, f"i-{purpose}-{len(self.released)}", "m1", gpu_name, gpus, 3.0, 0)
        self.db.set_machine(purpose, state="ready", ssh_host="203.0.113.7", ssh_port=40100)

    def advance(self, purpose):
        return self.get(purpose)["state"] if self.get(purpose) else "gone"

    def release(self, purpose, note="released"):
        if self.get(purpose):
            self.db.set_machine(purpose, state="gone", note=note)
            self.released.append(purpose)
            self.lost.discard(purpose)

    def machine(self, row):
        return Machine(row["name"], row["ssh_host"], row["ssh_port"], "/dev/null")

    def hand_over(self, purpose, public_key, deadline):
        m = Machine(purpose, "203.0.113.7", 40100, "/dev/null", None, "agent")
        m.host_key = lambda: "ssh-ed25519 SANDBOXKEY"
        return m

    def probe(self, purpose):
        return {"gpus": [{"name": "NVIDIA GeForce RTX 4090", "memory_gib": 24}] * 8, "cpus": 64, "memory": "500g",
                "disk_gib": 200, "network": "open (no allowlist)", "base_path": "/workspace/base",
                "work_path": "/workspace/work", "competition_repo": "/workspace/WeightSpaceProgramming"}

    def vanished(self, purpose):
        return purpose in self.lost

    def sample(self, purpose):
        return {"gpus": [{"util_pct": 97, "mem_used_mib": 20000, "mem_total_mib": 24564}] * 4,
                "note": "Screening directions on 203.0.113.7; best so far +2.1", "done": purpose in self.done}


class FakeRunner:
    name = "fake"
    """The 'agent': on kickoff it uploads its plan's edit spec through the staging protocol."""

    def __init__(self, private, tmp):
        self.private, self.tmp, self.kicked, self.saw_parent, self.methods = private, tmp, [], {}, {}

    def keep_alive(self):
        pass

    capacity = 6

    def prepare(self, sid, hotkey, busy=()):
        from axonry.agent.base import RunHandle
        return RunHandle(sid, f"axonry-{hotkey[:12].lower()}", f"chat-{sid}")

    def kickoff(self, h, message, bundle):
        import tarfile
        with tarfile.open(bundle) as tf:
            task = json.load(tf.extractfile("task.json"))
            assert "Deadline" in tf.extractfile("BRIEF.md").read().decode()
            cfg = tf.extractfile("ssh/config").read().decode()
            assert "HostName 203.0.113.7" in cfg and "Port 40100" in cfg and "User agent" in cfg
            assert tf.extractfile("ssh/known_hosts").read().decode() == "sandbox ssh-ed25519 SANDBOXKEY\n"
            method = tf.extractfile("submission/METHOD.md").read().decode()
            parent_files = {m.name[len("parent/edit/"):]: tf.extractfile(m).read() for m in tf.getmembers()
                            if m.name.startswith("parent/edit/")}
        n = method.split()[-1]
        self.methods[h.sid] = method
        self.saw_parent[h.sid] = bool(parent_files)
        out = self.tmp / h.sid
        out.mkdir(parents=True, exist_ok=True)
        (out / "edit.json").write_text(json.dumps(PLANS[n]["spec"]))
        (out / "REPORT.md").write_text(f"plan {n}")
        if parent_files:                                 # stacking: the spec names the chain, which is not uploaded
            spec = {**PLANS[n]["spec"], "parent": "parent/edit.json"}
            (out / "edit.json").write_text(json.dumps(spec))
        man = make_manifest(out)
        for f in man["files"]:
            self.private.put(task["r2"]["prefix"] + f["path"], (out / f["path"][5:]).read_bytes())
        self.private.put(task["r2"]["prefix"] + "manifest.json", json.dumps(man).encode())
        self.kicked.append(h.sid)

    def poll(self, h):
        from axonry.agent.base import RunStatus
        return RunStatus(False, Clock.t, tokens=12_345_678)

    def steer(self, h, m):
        pass

    def reject(self, h, a, r):
        pass

    def stop(self, h):
        pass

    def export(self, h, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "chat.json").write_text(json.dumps({"messages": [{"role": "user", "text": "go"}]}))
        return True


class FakeExecutor:
    """Evaluation machines: admission at once; an evaluation job finishes on the next look at it."""

    def __init__(self, tmp):
        self.tmp, self.specs, self.evaluated, self.jobs = tmp, {}, [], {}

    def admit(self, on, sid, spec_dir, base_doc):
        spec = json.loads((spec_dir / "edit.json").read_text())
        spec.pop("parent", None)
        self.specs[sid] = spec
        if "too_big" in spec["ops"]:
            return {"passed": False, "violations": ["layers.3.mlp.down_proj.weight: relative change 0.312 > 0.25"],
                    "info": {}, "model_digest": None}
        return {"passed": True, "violations": [], "info": {"tensors": 1}, "model_digest": "md-" + "".join(spec["ops"])}

    def start(self, on, model_id, base_doc, bench_docs, salt):
        self.evaluated.append(model_id)
        job = f"{on}/{model_id}/{len(self.evaluated)}"
        self.jobs[job] = {f"{b['id']}@{b['version']}": self.run_benchmark(model_id, b) for b in bench_docs}
        return job

    def result(self, on, job, model_id, bench_docs, salt):
        return self.jobs.pop(job)

    def run_benchmark(self, model_id, bench_doc):
        q = 0.0 if model_id.startswith("base:") else next(
            p["quality"] for p in PLANS.values() if p["spec"] == self.specs[model_id])
        rng = random.Random(f"{bench_doc['id']}")      # same items/seeds for every model (paired)
        rows = []
        for i in range(300):
            base_ok = rng.random() < 0.4
            extra = (i % 100) < q * 100
            for k in range(4):
                rows.append({"id": f"{bench_doc['id']}-{i}", "k": k, "correct": base_ok or extra})
        return "\n".join(json.dumps(r) for r in rows).encode()


def make(tmp_path):
    clock = Clock()
    pub, priv = LocalStore(tmp_path / "public", public=True), LocalStore(tmp_path / "private")
    chain = FakeChain()
    gh = FakeGitHub()
    db = DB(tmp_path / "db.sqlite")
    deps = Deps(db=db, chain=chain, github=gh, fleet=FakeFleet(db),
                issue_creds=lambda store, prefix, ttl: TempCreds("https://r2", "axonry-private", prefix,
                                                                 "AK", "SK", "ST",
                                                               clock() + ttl),
                runner=FakeRunner(priv, tmp_path / "agents"), executor=FakeExecutor(tmp_path),
                public=pub, private=priv, owner_keypair=OWNER, comp_rules=COMP_RULES,
                price_at=lambda ts: (500.0, {"x": 500.0, "y": 500.0}), scan_prs=gh.candidates,
                clock=clock, work_dir=tmp_path / "work")
    deps.registry_root = registry_copy(tmp_path, effective_block=0)   # not in force yet
    return Orchestrator(deps), chain, clock, pub


def open_rules(orch, block):
    """The rules take effect at `block` (their effective_block): the first intake reads from there."""
    from axonry.common.canon import write_json
    f = next((Path(orch.d.registry_root) / "rules" / BASE).glob("*.json"))
    write_json(f, {**json.loads(f.read_text()), "effective_block": block})
    orch.intake()


def registry_copy(tmp_path, **rules_patch):
    """The repo's registry with the current rules patched (None removes a key)."""
    import shutil
    from axonry.common.canon import write_json
    root = tmp_path / "registry"
    if not root.exists():
        shutil.copytree(registry.ROOT, root)
        for extra in sorted((root / "rules" / BASE).glob("*.json"))[1:]:   # the tests patch one rules version
            extra.unlink()
    rules_file = next((root / "rules" / BASE).glob("*.json"))
    doc = json.loads(rules_file.read_text())
    doc["fee"]["assets"] = ["tao", "alpha"]             # the tests pay in both
    doc.pop("open_block", None)                         # open from the first block unless a test says otherwise
    doc = {**doc, **rules_patch}
    write_json(rules_file, {k: v for k, v in doc.items() if v is not None})
    return root


BASE = "qwen35-0.8b-think"
RULES = registry.load_rules_versions(BASE)[-1]
RUN_USD = pricing.run_cost_musd(RULES) / pricing.MUSD
HOUR = pricing.hour_musd(RULES)                  # micro-USD of one hour of a run
FEE = RUN_USD * pricing.RAO // 500 + 1          # rao of TAO covering one whole run at 500 USD/TAO


def deposit(chain, n, block, amount=FEE, asset="tao"):
    chain.blocks.setdefault(block, []).append(Deposit(block, 0, COLDKEYS[n], asset, amount, Clock.t))


def pay(orch, chain, n, block, idx, parent=None, blobs=None):
    """A miner's whole submission: a deposit covering one fee, the content committed, the PR opened."""
    blobs = blobs or blobs_for(n, parent)
    deposit(chain, n, block)
    commit(chain, n, blobs, block, idx)
    number = len(orch.d.github.prs) + 1
    orch.d.github.add_pr(number, HOTKEYS[n], f"commit{n}{number}", blobs)
    return number


def commit(chain, n, blobs, block, idx=1, github_id=1):
    """Content committed on chain by n's hotkey (set_commitment: before every push)."""
    chain.blocks.setdefault(block, []).append(
        Commitment(block, idx, HOTKEYS[n], content_digest_of_blobs(blobs), github_id, Clock.t))


def run_until_quiet(orch, clock, rounds=12):
    for _ in range(rounds):
        orch.tick()
        clock.t += 9 * 3600     # past every deadline + grace


def test_full_flow(tmp_path, capsys):
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    w0 = json.loads(pub.get(layout.WEIGHTS))
    assert w0["weights"] == [{"uid": 0, "hotkey": None, "weight": 1.0}]   # no champion yet: burn

    for i, n in enumerate("ABCD"):
        pay(orch, chain, n, 101 + i, 2)
    chain.head = 110
    open_rules(orch, 101)
    assert [s["status"] for s in orch.db.subs()] == ["QUEUED"] * 4   # committed + approved

    run_until_quiet(orch, clock)
    st = {s["hotkey"]: (s["status"], s["reason_code"]) for s in orch.db.subs()}
    assert st[HOTKEYS["A"]] == ("PUBLISHED", "NEW_CHAMPION")
    assert st[HOTKEYS["B"]] == ("REJECTED", "REJECTED_RULES")
    assert st[HOTKEYS["C"]] == ("PUBLISHED", "NOT_DOMINATING")   # same weights as A, decided after A
    assert st[HOTKEYS["D"]] == ("PUBLISHED", "NOT_DOMINATING")

    w = json.loads(pub.get(layout.WEIGHTS))
    assert w["weights"] == [{"uid": UIDS[HOTKEYS["A"]], "hotkey": HOTKEYS["A"], "weight": 1.0}]
    assert validator.check(w, OWNER.ss58_address, NETUID, 0) is None
    assert validator.resolve(w, {UIDS[HOTKEYS["A"]]: HOTKEYS["A"]}) == {UIDS[HOTKEYS["A"]]: 1.0}

    sid_a, sid_b = "0000000101-0002", "0000000102-0002"
    score = json.loads(pub.get(layout.submission(sid_a, "score.json")))
    assert score["dominates_champion"] and score["wins"] >= 2
    vs = score["benchmarks"]["math500@1"]["vs_champion"]        # the match can be redone from this file alone
    assert {"ref", "ref_acc", "d", "se", "tie_ok", "win", "tie_at", "win_at"} <= set(vs)
    assert set(vs["params"]) >= {"eps_tie", "delta_win"} and score["judge"]["n_win"] >= 1
    assert score["generated"]["salt"].startswith("axonry-") and score["rules_sha256"]
    assert json.loads(pub.get(layout.submission(sid_a, "submission.json")))["coldkey"] == COLDKEYS["A"]
    adm = json.loads(pub.get(layout.submission(sid_b, "admission.json")))
    assert not adm["passed"] and "relative change" in adm["violations"][0]
    import gzip
    row = json.loads(gzip.decompress(pub.get(layout.results(sid_a, "math500@1"))).splitlines()[0])
    assert set(row) == {"id", "k", "correct"}                 # no generated text in public results
    assert pub.get(layout.submission_agent_prefix(sid_a) + "chat.json")
    assert pub.get(layout.submission_edit_prefix(sid_a) + "edit.json")
    sub_b = json.loads(pub.get(layout.submission(sid_b, "submission.json")))
    assert sub_b["status"] == "REJECTED" and sub_b["reason_code"] == "REJECTED_RULES"

    gh = orch.d.github
    assert set(gh.tags) == {sid_a, "0000000104-0002", sid_b, "0000000103-0002"}   # every run was locked
    assert gh.merged == [1]                                   # only the champion's PR is merged
    assert {n for n, _ in gh.comments} == {1, 2, 3, 4}        # every finished PR gets its result and is closed
    assert all(gh.prs[n]["state"] == "closed" for n in (1, 2, 3, 4))

    # the miner CLI reads the same public files (no keys)
    from axonry.miner.board import cmd_board
    reader = miner.PublicReader(store=pub)
    orch.db.set("status_at", 0)
    orch.publish_status()
    cmd_board(reader, color=False)
    out = capsys.readouterr().out
    assert sid_a in out and "NEW CHAMPION" in out
    cmd_board(reader, color=False, miner=sid_b)
    out = capsys.readouterr().out
    assert "REJECTED_RULES" in out and "relative change 0.312 > 0.25" in out
    cmd_board(reader, color=False, miner=sid_a)
    out = capsys.readouterr().out
    assert "NEW_CHAMPION" in out and "win" in out and "ACCOUNT" in out

    # nothing secret at rest, nothing left in the private bucket once tickets are finished
    task = orch.db.run(sid_a)["task"]
    assert "private_key" not in task["sandbox"] and "secret_access_key" not in task["r2"]
    priv = orch.d.private
    assert not list(priv.list("tasks/"))
    orch.db.set("janitor_at", 0)
    orch.janitor()
    assert not list(priv.list("staging/"))


def test_accounts_and_fees(tmp_path):
    """Deposits of any size, at any time, accumulate in the account of the coldkey that sent them, each in USD at the
    price of its own block. An entry is reviewed only when the account of the coldkey owning the hotkey covers a whole
    run (hours x GPUs x the GPU-hour price); without a review fee in the rules nothing is charged before the run. The run is charged each full hour and settled when
    it ends (here: an early finish after 2.5 h, which also frees the GPU at once)."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    v1 = blobs_for("A")
    commit(chain, "A", v1, 101)
    deposit(chain, "A", 101, FEE // 2)
    deposit(chain, "B", 100, 3 * FEE)                  # another coldkey's money never pays for A's hotkey
    gh.add_pr(1, HOTKEYS["A"], "a1", v1)
    chain.head = 101
    open_rules(orch, 100)
    sid, ck = "0000000101-0001", COLDKEYS["A"]
    s = orch.db.sub(sid)
    assert (s["status"], s["reason_code"], s["reason"]) == ("WAITING", "INSUFFICIENT_FUNDS", "insufficient funds")
    assert abs(orch.db.accounts()[ck]["balance_usd"] - RUN_USD / 2) < 0.01

    orch.d.price_at = lambda ts: (1000.0, {"x": 1000.0, "y": 1000.0})    # TAO doubles: only later deposits gain
    deposit(chain, "A", 130, FEE // 2)
    chain.head = 130
    orch.intake()
    a = orch.db.accounts()[ck]
    assert orch.db.sub(sid)["status"] == "REVIEW" and abs(a["balance_usd"] - 1.5 * RUN_USD) < 0.01 and a["charged_usd"] == 0
    gh.verdicts[(1, content_digest_of_blobs(v1))] = "approved"
    orch.intake()
    s = orch.db.sub(sid)
    a = orch.db.accounts()[ck]
    assert s["status"] == "QUEUED" and s["coldkey"] == ck and s["record"]["fee"]["max_usd"] == RUN_USD
    assert a["charged_usd"] == 0 and a["held_usd"] == RUN_USD and abs(a["available_usd"] - RUN_USD / 2) < 0.01

    orch.schedule()
    kick = clock.t
    clock.t = kick + int(2.5 * 3600)
    orch.supervise()
    assert orch.db.charged(sid) == {1: HOUR, 2: HOUR}                               # each full hour
    assert orch.db.accounts()[ck]["held_usd"] == RUN_USD - 2 * HOUR / pricing.MUSD
    orch.db.set("status_at", 0)
    orch.publish_status()                               # a miner's view: its balance and its run, together
    m = next(x for x in json.loads(pub.get(layout.STATUS))["miners"] if x["hotkey"] == HOTKEYS["A"])
    assert m["account"]["held_usd"] == RUN_USD - 2 * HOUR / pricing.MUSD and m["run"]["sid"] == sid
    assert m["run"]["note"] == "Screening directions on [redacted]; best so far +2.1"   # the run's secrets removed
    assert [g["util_pct"] for g in m["run"]["gpus"]] == [97] * 4 and m["run"]["sampled_at"] == clock.t
    assert m["run"]["agent"]["tokens_m"] == 12.35                                # the agent's tokens so far
    assert m["run"]["gpu_usd"] == round(pricing.used_musd(RULES, clock.t - kick) / 1e6, 2)   # its GPUs so far
    from axonry.miner.board import board, detail
    reader = miner.PublicReader(store=pub)
    st_doc = reader.get(layout.STATUS)
    over = board(st_doc, None, color=False)                                     # the overview: time left, status
    assert "3h30m left · Screening directions" in over and "GPU" not in over.split("2. Not in queue")[0]
    text = detail(reader, st_doc, None, str(m["uid"]), color=False)            # one miner in full
    assert "RUN (live" in text and "GPU 3" in text and " 97%" in text and "best so far +2.1" in text
    clock.t += 300
    orch.d.fleet.done.add(sid)                          # the agent ran axonry-done (a complete upload exists)
    orch.supervise()
    s = orch.db.sub(sid)
    used = pricing.used_musd(RULES, clock.t - kick)
    assert s["status"] == "COLLECTING" and s["record"]["agent_outcome"] == "finished"
    assert s["record"]["run"]["gpu_usd"] == round(used / 1e6, 2) and s["record"]["run"]["agent"]["tokens_m"] == 12.35
    assert s["record"]["run"]["note"].startswith("Screening directions")      # the run's last view is kept
    assert orch.d.fleet.get(sid) is None and sid in orch.d.fleet.released           # the machine is released
    assert sum(orch.db.charged(sid).values()) == used and s["record"]["fee"]["charged_usd"] == used / 1e6
    a = orch.db.accounts()[ck]
    assert a["held_usd"] == 0 and abs(a["balance_usd"] - (1.5 * RUN_USD - used / 1e6)) < 0.01
    orch.supervise()                                    # charged once, however often it looks again
    assert sum(orch.db.charged(sid).values()) == used
    orch.db.set("status_at", 0)
    orch.publish_status()
    assert json.loads(pub.get(layout.STATUS))["accounts"][ck]["charged_usd"] == used / 1e6
    assert next(x for x in json.loads(pub.get(layout.STATUS))["miners"] if x["hotkey"] == HOTKEYS["A"])["run"] is None


def test_one_registration_one_entry(tmp_path):
    """Only a hotkey's first commitment counts: a later one (before or after its run) is void, its PRs carry
    nothing, and nothing more is ever charged."""
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    gh = orch.d.github
    v1, v2 = blobs_for("A"), blobs_for("A") | {"METHOD.md": b"second method of A"}
    deposit(chain, "A", 101, 3 * FEE)
    commit(chain, "A", v1, 101)
    commit(chain, "A", v2, 102)                         # a second commitment while the first waits: void
    gh.add_pr(1, HOTKEYS["A"], "a2", v2)
    chain.head = 102
    open_rules(orch, 101)
    (s,) = orch.db.subs()
    assert s["sid"] == "0000000101-0001" and s["status"] == "WAITING" and s["reason_code"] == "NO_PR"
    assert orch.db.get("prs")["1"]["reason_code"] == "UNCOMMITTED" and "neither the entry's content" in orch.db.get("prs")["1"]["reason"]
    gh.push(1, "a1", v1)                                # the PR now carries the committed content
    orch.intake()
    run_until_quiet(orch, clock)
    assert orch.db.sub(s["sid"])["status"] == "PUBLISHED"
    commit(chain, "A", v2, 200)                         # after the run: still void
    gh.add_pr(2, HOTKEYS["A"], "a3", v2)
    chain.head = 200
    orch.intake()
    run_until_quiet(orch, clock)
    assert len(orch.db.subs()) == 1 and orch.db.get("prs")["2"]["reason_code"] == "PARTICIPATED"
    assert "1" not in orch.db.get("prs")                # its own PR is shown with the submission, not again
    assert orch.db.accounts()[COLDKEYS["A"]]["charged_usd"] == pricing.gpu_max_musd(RULES) / 1e6   # GPUs to the deadline


def test_copy_cannot_overtake_the_original(tmp_path):
    """B copies A's method, slightly altered (an exact copy cannot even enter: A's commitment owns that digest), after
    A committed it. B is reviewed and runs first, but decisions follow the chain order of the commitments: A is
    decided first and becomes champion; B, judged against A, cannot dominate."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    copy = {"entry.json": blobs_for("B")["entry.json"], "METHOD.md": b"reworded method of A"}
    na = pay(orch, chain, "A", 101, 1)
    nb = pay(orch, chain, "B", 105, 1, blobs=copy)
    chain.head = 105
    open_rules(orch, 101)
    sa, sb = "0000000101-0001", "0000000105-0001"
    gh.verdicts[(nb, content_digest_of_blobs(copy))] = "approved"    # the copy is reviewed first
    run_until_quiet(orch, clock, rounds=4)
    assert orch.db.sub(sb)["status"] == "SCORED"                   # finished, but waits for the earlier priority
    assert orch.db.sub(sa)["status"] == "REVIEW"
    gh.verdicts[(na, content_digest_of_blobs(blobs_for("A")))] = "approved"
    run_until_quiet(orch, clock, rounds=6)
    assert orch.db.sub(sa)["reason_code"] == "NEW_CHAMPION"
    assert orch.db.sub(sb)["reason_code"] == "NOT_DOMINATING"
    assert orch.db.champion(BASE)["sid"] == sa


def test_nobody_else_can_carry_a_hotkeys_entry(tmp_path):
    """Mallory opens a PR with A's committed content in A's directory: it carries nothing (A's commitment names A's
    GitHub id), so it can neither charge A nor hold A's entry hostage; A's own PR carries it."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    deposit(chain, "A", 101)
    commit(chain, "A", blobs_for("A"), 101)
    gh.add_pr(1, HOTKEYS["A"], "evil", blobs_for("A"))
    gh.prs[1]["author"] = "mallory"
    chain.head = 101
    open_rules(orch, 101)
    (s,) = orch.db.subs()
    assert s["status"] == "WAITING" and s["reason_code"] == "NO_PR" and not orch.db.accounts()[COLDKEYS["A"]]["charged_usd"]
    w = orch.db.get("prs")["1"]
    assert w["reason_code"] == "UNCOMMITTED" and "mallory" in w["reason"]
    gh.add_pr(2, HOTKEYS["A"], "mine", blobs_for("A"))
    orch.intake()
    s = orch.db.sub(s["sid"])
    assert s["status"] == "QUEUED" and s["pr_number"] == 2 and orch.db.get("prs")["1"]["reason_code"] == "PARTICIPATED"


def test_a_content_belongs_to_its_earliest_commitment(tmp_path):
    """An exact copy is rejected at its commitment; another hotkey's content in my directory and a reused GitHub
    login carry nothing; nothing here depends on the reviewer."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    a = blobs_for("A")
    for i, n in enumerate("ABC"):
        deposit(chain, n, 98 + i)
    commit(chain, "A", a, 101)
    commit(chain, "B", a, 102)                            # B saw A's PR and committed the same content later
    commit(chain, "C", blobs_for("C"), 103)
    gh.add_pr(1, HOTKEYS["B"], "copy", a)
    gh.add_pr(2, HOTKEYS["A"], "c-in-a", blobs_for("C"))   # C's content in A's directory
    gh.add_pr(3, HOTKEYS["A"], "a-renamed", a)            # a new account took the login "u" after a rename
    gh.prs[3]["author_id"] = 777
    chain.head = 103
    open_rules(orch, 98)
    sb = orch.db.of_hotkey(HOTKEYS["B"])
    assert (sb["status"], sb["reason_code"]) == ("REJECTED", "COPY") and "0000000101-0001" in sb["reason"]
    w = orch.db.get("prs")
    assert w["1"]["reason_code"] == "PARTICIPATED"
    assert w["2"]["reason_code"] == "UNCOMMITTED" and "neither the entry's content" in w["2"]["reason"]
    assert w["3"]["reason_code"] == "UNCOMMITTED" and "777" in w["3"]["reason"]
    assert orch.db.of_hotkey(HOTKEYS["A"])["reason_code"] == "NO_PR"
    assert not any(orch.db.accounts()[COLDKEYS[n]]["charged_usd"] for n in "AB")


def test_disqualified_champion_hands_back_the_title(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    pay(orch, chain, "A", 101, 1)
    chain.head = 101
    open_rules(orch, 101)
    run_until_quiet(orch, clock)
    sid = orch.db.champion(BASE)["sid"]
    orch.disqualify(sid, "agent log: downloaded the benchmark")
    assert orch.db.champion(BASE) is None and orch.db.sub(sid)["status"] == "DISQUALIFIED"
    assert json.loads(pub.get(layout.WEIGHTS))["weights"][0]["uid"] == 0


def test_review_lock_and_carrier(tmp_path):
    """A review asking for changes sends the entry back to WAITING and, without a review fee in the rules, costs nothing. An approved one is locked at that commit and charged;
    the PR is only the carrier: a rebase keeps the approval, pushes after the lock never change the entry."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    na = pay(orch, chain, "A", 101, 1)
    nb = pay(orch, chain, "B", 102, 1)
    chain.head = 102
    open_rules(orch, 101)
    sa, sb = "0000000101-0001", "0000000102-0001"
    assert orch.db.sub(sa)["status"] == orch.db.sub(sb)["status"] == "REVIEW"

    gh.verdicts[(na, content_digest_of_blobs(blobs_for("A")))] = "changes"
    orch.intake()
    s = orch.db.sub(sa)
    assert (s["status"], s["reason_code"]) == ("WAITING", "REVIEW_CHANGES")
    assert s["reason"].startswith("reviewer note (full review on PR #") and "Only deletions" in s["reason"]
    assert not orch.db.accounts()[COLDKEYS["A"]]["charged_usd"]

    gh.verdicts[(nb, content_digest_of_blobs(blobs_for("B")))] = "approved"
    gh.prs[nb]["mergeable"] = False
    orch.intake()
    assert orch.db.sub(sb)["reason_code"] == "CONFLICT" and not orch.db.accounts()[COLDKEYS["B"]]["charged_usd"]
    gh.push(nb, "b-rebased", blobs_for("B"))
    gh.prs[nb]["mergeable"] = True
    orch.intake()
    s = orch.db.sub(sb)
    assert s["status"] == "QUEUED" and gh.tags[sb] == "b-rebased" and orch.db.accounts()[COLDKEYS["B"]]["held_usd"] == RUN_USD
    gh.push(nb, "b-later", blobs_for("B") | {"METHOD.md": b"later method of B"})
    orch.intake()
    orch.schedule()
    s = orch.db.sub(sb)
    assert s["status"] == "RUNNING" and s["locked_commit"] == "b-rebased" and orch.d.runner.methods[sb] == "method of B"


def test_stacking_on_a_published_parent(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    pay(orch, chain, "A", 101, 1)
    chain.head = 101
    open_rules(orch, 101)
    run_until_quiet(orch, clock, rounds=6)
    sa = "0000000101-0001"
    assert orch.db.sub(sa)["reason_code"] == "NEW_CHAMPION"

    pay(orch, chain, "E", 110, 1, parent=sa)
    chain.head = 110
    orch.intake()
    run_until_quiet(orch, clock, rounds=6)
    se = "0000000110-0001"
    assert orch.d.runner.saw_parent[se]                       # the parent's edit was in the bundle
    assert orch.db.sub(se)["record"]["parent"] == sa
    assert orch.db.sub(se)["reason_code"] == "NEW_CHAMPION"   # judged against the champion (its parent)
    assert pub.get(layout.submission_edit_prefix(se) + "edit.json")
    assert pub.get(layout.submission_edit_prefix(se) + "parent/edit.json") is None   # chains are not stored twice
    chain = tmp_path / "chain"
    orch._materialize_chain(se, chain)                        # rebuilt from the public edits, nested by parent
    assert json.loads((chain / "edit.json").read_text())["parent"] == "parent/edit.json"
    assert (chain / "parent" / "edit.json").exists() and not (chain / "parent" / "parent").exists()


def test_a_parent_that_can_never_qualify_rejects(tmp_path):
    """A parent that is not a submission at all can never become one: the entry is rejected, uncharged."""
    orch, chain, clock, pub = make(tmp_path)
    pay(orch, chain, "E", 101, 1, parent="0000000001-0001")
    chain.head = 101
    open_rules(orch, 101)
    (s,) = orch.db.subs()
    assert (s["status"], s["reason_code"]) == ("REJECTED", "BAD_PARENT") and not orch.db.accounts()[COLDKEYS["E"]]["charged_usd"]


def test_our_outages_never_fail_a_ticket(tmp_path):
    """An unreachable evaluation host backs off without counting; only the submission's own failing jobs count."""
    from axonry.evaluation.executor import ExecError, JobFailed
    orch, chain, clock, pub = make(tmp_path)
    pay(orch, chain, "A", 101, 1)
    chain.head = 101
    open_rules(orch, 101)
    sid = "0000000101-0001"
    real_admit = orch.d.executor.admit
    outages = {"n": 0}

    def flaky_admit(*a):
        outages["n"] += 1
        if outages["n"] <= 5:
            raise ExecError("eval host: no answer within 60s")
        return real_admit(*a)
    orch.d.executor.admit = flaky_admit
    for _ in range(12):
        orch.tick()
        clock.t += 3 * 3600                     # past deadlines and any backoff
    assert orch.db.sub(sid)["status"] == "PUBLISHED" and orch.db.sub(sid)["infra_retries"] == 0
    assert orch.db.sub(sid)["reason"] is None                        # no stale note of an outage once it passed

    pay(orch, chain, "D", 120, 1)
    chain.head = 120
    orch.intake()
    orch.d.executor.admit = lambda *a: (_ for _ in ()).throw(JobFailed("job exited 1: model crashed"))
    for _ in range(10):
        orch.tick()
        clock.t += 3 * 3600
    s = orch.db.sub("0000000120-0001")
    assert s["status"] == "FAILED" and s["reason_code"] == "EVAL_ERROR"


def test_board(tmp_path):
    """axonry-miner board from public files only: champion + queue, not in queue (and why), recent matches with
    the not-worse / win thresholds; a deregistered champion shows its weight burned."""
    from axonry.miner.board import board
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    for i, n in enumerate("ACD"):
        pay(orch, chain, n, 101 + i, 2)
    chain.head = 110
    open_rules(orch, 101)
    run_until_quiet(orch, clock)                                   # A champion; C, D decided against A
    gh = orch.d.github
    gh.approve_all = False
    pay(orch, chain, "B", 120, 1)                                  # B is funded and waits for review
    gh.add_pr(9, HOTKEYS["E"], "e1", blobs_for("E"))               # E pushed without committing
    gh.prs[9]["title"] = "Loop removal\x1b[31m v2"                 # a title trying to inject terminal codes
    deposit(chain, "F", 121)                                       # F: registered, funds, no commitment
    gh.prs[3]["state"] = "closed"
    chain.head = 121
    orch.intake()
    orch.db.set("status_at", 0)
    orch.publish_status()
    reader = miner.PublicReader(store=pub)
    status = reader.get(layout.STATUS)
    line = reader.get(layout.line(BASE))
    text = board(status, line, n=20, color=False)
    print(board(status, line, n=20, color=True))
    sid_a = orch.db.champion(BASE)["sid"]
    t1, t2, t3 = text.split("\n\n")[:3]
    assert "CHAMPION" in t1 and sid_a in t1 and "in review" in t1
    assert "not committed" in t2 and "\x1b" not in t2 and "account: " in t2 and HOTKEYS["F"][:10] in t2
    assert "NEW CHAMPION" in t3 and "[" in t3 and "math500" in t3
    a_row = next(r for r in line["entries"] if r["sid"] == sid_a)
    acc, tie_at, win_at, verdict = a_row["scores"]["math500@1"]
    assert tie_at < win_at and verdict in ("win", "tie")
    rows3 = t3.splitlines()[3:]
    assert len(rows3) == 3 and len({ln.index("[") for ln in rows3}) == 1      # columns aligned
    assert HOTKEYS["D"][:10] not in t2               # D took part already: not listed as a funded newcomer
    UIDS.pop(HOTKEYS["A"])                                            # the champion is deregistered
    try:
        orch.db.set("status_at", 0)
        orch.publish_status()
        assert "weight burned" in board(reader.get(layout.STATUS), line, color=False).split("\n\n")[0]
    finally:
        UIDS[HOTKEYS["A"]] = 1


def test_a_vanished_machine_runs_again_at_no_cost(tmp_path):
    """vast drops a run's machine mid-run: our failure. The agent is stopped, everything charged is refunded, the
    staging area is cleared, and the submission keeps its priority and runs again from the start on a new machine.
    Runs never hold more machines than max_concurrent."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.max_concurrent = 1
    for n, (name, block) in enumerate((("A", 101), ("B", 102))):
        v = blobs_for(name)
        commit(chain, name, v, block)
        deposit(chain, name, block, 2 * FEE)
        orch.d.github.add_pr(n + 1, HOTKEYS[name], f"{name.lower()}1", v)
    chain.head = 102
    open_rules(orch, 101)
    sa, sb = "0000000101-0001", "0000000102-0001"
    orch.schedule()
    assert orch.db.sub(sa)["status"] == "RUNNING" and orch.db.sub(sb)["status"] == "QUEUED"
    assert [m["name"] for m in orch.d.fleet.active()] == [sa]                        # one machine at a time
    clock.t += int(1.5 * 3600)
    orch.supervise()
    assert orch.db.charged(sa) == {1: HOUR}
    orch.d.fleet.lost.add(sa)
    clock.t += 60
    orch.schedule()                                     # noticed, refunded, and on a new machine at once
    s = orch.db.sub(sa)
    assert orch.d.runner.kicked.count(sa) == 2 and orch.d.fleet.released == [sa]
    assert s["status"] == "RUNNING" and s["record"]["timeline"]["kickoff"] == clock.t and orch.db.charged(sa) == {}
    assert orch.db.sub(sb)["status"] == "QUEUED"                                        # priority kept
    assert orch.db.accounts()[COLDKEYS["A"]]["held_usd"] == RUN_USD
    run_until_quiet(orch, clock)                        # the run that was lost once still takes the title
    assert orch.db.champion(BASE)["sid"] == sa and 1 in orch.d.github.merged


def test_a_failed_start_keeps_the_machine(tmp_path):
    """Starting a run failed for a reason of ours (the agent backend, R2, a dropped connection): the run is retried on
    the same machine after the backoff; only a machine that fails three starts in a row is replaced."""
    orch, chain, clock, pub = make(tmp_path)
    v = blobs_for("A")
    commit(chain, "A", v, 101)
    deposit(chain, "A", 101, 2 * FEE)
    orch.d.github.add_pr(1, HOTKEYS["A"], "a1", v)
    chain.head = 101
    open_rules(orch, 101)
    sid, real = "0000000101-0001", orch.d.runner.prepare
    orch.d.runner.prepare = lambda *a: (_ for _ in ()).throw(RuntimeError("arbos relay unreachable"))
    orch.schedule()
    assert orch.db.sub(sid)["status"] == "QUEUED" and orch.d.fleet.get(sid)["state"] == "ready"   # kept
    orch.d.runner.prepare = real
    clock.t += 120
    orch.schedule()
    assert orch.db.sub(sid)["status"] == "RUNNING" and orch.d.fleet.released == []


def test_the_queue_waits_when_no_machine_is_free(tmp_path):
    """All owned hosts busy and vast.ai disabled: the submission stays queued, in its place, with nothing held against
    it, and starts as soon as a machine is free."""
    from axonry.pool import NoMachine
    orch, chain, clock, pub = make(tmp_path)
    v = blobs_for("A")
    commit(chain, "A", v, 101)
    deposit(chain, "A", 101, 2 * FEE)
    orch.d.github.add_pr(1, HOTKEYS["A"], "a1", v)
    chain.head = 101
    open_rules(orch, 101)
    sid, real = "0000000101-0001", orch.d.fleet.rent
    orch.d.fleet.rent = lambda *a: (_ for _ in ()).throw(NoMachine("no owned machine free"))
    orch.schedule()
    s = orch.db.sub(sid)
    assert s["status"] == "QUEUED" and s["reason"] is None and not orch.db.get(f"backoff:{sid}:n")
    orch.d.fleet.rent = real
    orch.schedule()
    assert orch.db.sub(sid)["status"] == "RUNNING"


def test_a_vanished_evaluation_machine_is_replaced(tmp_path):
    from axonry.pool.machine import EVAL
    orch, chain, clock, pub = make(tmp_path)
    v = blobs_for("A")
    commit(chain, "A", v, 101)
    deposit(chain, "A", 101, 2 * FEE)
    orch.d.github.add_pr(1, HOTKEYS["A"], "a1", v)
    chain.head = 101
    open_rules(orch, 101)
    orch.schedule()
    clock.t += 9 * 3600
    orch.supervise()                                   # COLLECTING: work for the evaluation machine
    orch.eval_machine()
    assert orch.d.fleet.get(EVAL)["state"] == "ready"
    orch.d.fleet.lost.add(EVAL)
    orch.eval_machine()
    assert EVAL in orch.d.fleet.released and orch.d.fleet.get(EVAL) is None
    orch.d.fleet.lost.clear()
    orch.eval_machine()                                # work remains: a new one
    assert orch.d.fleet.get(EVAL)["state"] == "ready"


def test_the_evaluation_machine_is_kept_while_a_run_waits_to_be_collected(tmp_path):
    from axonry.pool.machine import EVAL
    orch, chain, clock, pub = make(tmp_path)
    orch.d.eval_idle_minutes = 1
    v = blobs_for("A")
    commit(chain, "A", v, 101)
    deposit(chain, "A", 101, 2 * FEE)
    orch.d.github.add_pr(1, HOTKEYS["A"], "a1", v)
    chain.head = 101
    open_rules(orch, 101)
    orch.schedule()
    clock.t = orch.db.run("0000000101-0001")["task"]["run"]["deadline"]
    orch.supervise()                                   # deadline: COLLECTING
    orch.eval_machine()
    for _ in range(3):                                 # well past eval_idle_minutes, still to be collected
        clock.t += 120
        orch.eval_machine()
    assert orch.d.fleet.get(EVAL)["state"] == "ready" and EVAL not in orch.d.fleet.released


def test_the_first_intake_starts_at_the_first_rules_effective_block(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    chain.head = 120
    with pytest.raises(ValueError, match="effective_block"):     # the rules not in force yet (block 0)
        orch.intake()
    orch.d.registry_root = registry_copy(tmp_path, effective_block=105)
    read, items = [], chain.items_in_block
    chain.items_in_block = lambda b, *a: read.append(b) or items(b, *a)
    orch.intake()
    assert read == list(range(105, 121)) and orch.db.get("cursor") == 120


def test_started_before_the_first_rules_it_waits(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path, effective_block=500)
    chain.head = 120
    orch.intake()
    orch.bootstrap()
    orch.publish_status(every=0)
    assert pub.get(layout.WEIGHTS) is None and orch.db.get("cursor") == 499


def test_a_new_base_model_starts_its_own_line_and_the_old_one_stays(tmp_path):
    """Rules moving to another base model: from their block that base's line is in force (no champion yet: the weight
    is burned), the first line's champion, history and scores stay published, and index.json lists both."""
    from axonry.common.canon import write_json
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    pay(orch, chain, "A", 101, 2)
    chain.head = 101
    open_rules(orch, 101)
    run_until_quiet(orch, clock)
    sid_a = orch.db.champion(BASE)["sid"]
    root = Path(orch.d.registry_root)
    other = {**json.loads((root / "bases" / f"{BASE}.json").read_text()), "id": "other-base"}
    write_json(root / "bases" / "other-base.json", other)
    rules = json.loads(next((root / "rules" / BASE).glob("*.json")).read_text())
    (root / "rules" / "other-base").mkdir()
    write_json(root / "rules" / "other-base" / "v1.json", {**rules, "base": "other-base", "effective_block": 130})
    orch.bootstrap()                                    # a new rules version comes with a deploy (a restart)
    orch.refresh_weights()
    assert json.loads(pub.get(layout.WEIGHTS))["base"] == BASE                     # staged, not in force yet
    chain.head = 131
    orch.intake()
    orch.refresh_weights()
    w = json.loads(pub.get(layout.WEIGHTS))
    assert w["base"] == "other-base" and w["weights"] == [{"hotkey": None, "uid": 0, "weight": 1.0}]
    idx = json.loads(pub.get(layout.INDEX))
    assert idx["base"] == "other-base" and [v["base"] for v in idx["versions"]] == [BASE, "other-base"]
    assert all(pub.get(v["rules"]) for v in idx["versions"])
    old = json.loads(pub.get(layout.line(BASE)))
    assert old["champion"]["sid"] == sid_a and [h["sid"] for h in old["history"]] == [sid_a]
    assert next(r for r in old["entries"] if r["sid"] == sid_a)["decision"] == "NEW_CHAMPION"


def test_every_figure_of_a_decision_is_kept_for_good(tmp_path):
    """A decided entry's score, record and per-question results, and its reference's (the base model's), are all
    public and written once: a match can be recomputed later from the public bucket alone."""
    from axonry.common.store import ImmutableObjectError
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    pay(orch, chain, "A", 101, 2)
    chain.head = 101
    open_rules(orch, 101)
    run_until_quiet(orch, clock)
    sid = orch.db.champion(BASE)["sid"]
    score = json.loads(pub.get(layout.score(sid)))
    assert score["model"] == sid and all(b["vs_champion"]["ref"] == f"base:{BASE}" for b in score["benchmarks"].values())
    assert json.loads(pub.get(layout.submission(sid, "submission.json")))["decision"] == "NEW_CHAMPION"
    base_score = json.loads(pub.get(layout.score(f"base:{BASE}")))
    assert base_score["model"] == f"base:{BASE}" and set(base_score["benchmarks"]) == set(RULES.benchmark_keys())
    for k in RULES.benchmark_keys():
        assert pub.get(layout.results(sid, k)) and pub.get(layout.results(f"base:{BASE}", k))
    line = json.loads(pub.get(layout.line(BASE)))
    assert line["base_model"]["accuracy"] == {k: b["acc"] for k, b in base_score["benchmarks"].items()}
    with pytest.raises(ImmutableObjectError):
        pub.put(layout.results(f"base:{BASE}", "math500@1"), b"changed")
def test_commitments_before_the_opening_block_count_for_nothing(tmp_path):
    """A miner's commitment before open_block is void and its first one from open_block on is its entry. The owner's
    hotkey (burned weight) is bound by neither rule."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path, effective_block=100, open_block=110)
    commit(chain, "A", blobs_for("A"), 101)
    chain.blocks.setdefault(102, []).append(
        Commitment(102, 1, OWNER.ss58_address, content_digest_of_blobs(blobs_for("B")), 1, Clock.t))
    chain.head = 105
    orch.intake()
    assert [s["hotkey"] for s in orch.db.subs()] == [OWNER.ss58_address]
    commit(chain, "A", blobs_for("A"), 111)
    chain.blocks[112] = [Commitment(112, 1, OWNER.ss58_address, content_digest_of_blobs(blobs_for("C")), 1, Clock.t)]
    chain.head = 112
    orch.intake()
    assert orch.db.of_hotkey(HOTKEYS["A"])["sid"] == "0000000111-0001"
    assert [s["sid"] for s in orch.db.subs() if s["hotkey"] == OWNER.ss58_address] == [
        "0000000102-0001", "0000000112-0001"]


def test_a_pr_opened_before_its_commitment_carries_nothing(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    gh = orch.d.github
    deposit(chain, "A", 101, FEE)
    commit(chain, "A", blobs_for("A"), 101)
    gh.add_pr(1, HOTKEYS["A"], "a1", blobs_for("A"))
    gh.prs[1]["opened_at"] = int(Clock.t) - 60          # pushed before the commitment: anyone could have copied it
    chain.head = 101
    open_rules(orch, 101)
    (s,) = orch.db.subs()
    assert s["reason_code"] == "NO_PR" and "opened before" in orch.db.get("prs")["1"]["reason"]


def _unseen_a_public_b(orch, chain):
    """A commits at block 101 and opens no PR; B commits at 105 and opens its PR (B's content is public)."""
    deposit(chain, "A", 101)
    commit(chain, "A", blobs_for("A"), 101)
    nb = pay(orch, chain, "B", 105, 1)
    chain.head = 105
    open_rules(orch, 101)
    return "0000000101-0001", "0000000105-0001", nb


def test_without_a_reveal_deadline_a_late_reveal_goes_first(tmp_path):
    """The case reveal_hours closes. Without it, a commitment can stay unseen for as long as its owner likes: A opens
    its PR a month after B's content became public and still goes ahead of B in the review queue (so B is compared
    with A, not A with B) and in decisions."""
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    gh = orch.d.github
    gh.approve_all = False                              # both stay in review: the review queue shows the order
    sa, sb, nb = _unseen_a_public_b(orch, chain)
    assert orch.db.sub(sa)["reason_code"] == "NO_PR" and orch.db.sub(sb)["status"] == "REVIEW"
    clock.t = Clock.t + 30 * 24 * 3600
    gh.add_pr(nb + 1, HOTKEYS["A"], "a1", blobs_for("A"))
    gh.prs[nb + 1]["opened_at"] = clock.t
    orch.intake()
    orch.publish_status()
    assert [r["sid"] for r in json.loads(pub.get(layout.STATUS))["review"]] == [sa, sb]


def test_a_commitment_not_revealed_in_time_is_void(tmp_path):
    """With reveal_hours, a commitment whose content no PR carries within that time of its block is void
    (REVEAL_EXPIRED, final): a PR opened afterwards carries nothing, and the public entry after it keeps its place."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path, reveal_hours=24)
    orch.bootstrap()
    gh = orch.d.github
    gh.approve_all = False
    sa, sb, nb = _unseen_a_public_b(orch, chain)
    s = orch.db.sub(sa)
    assert s["reason_code"] == "NO_PR" and "or the entry is void" in s["reason"]
    clock.t = Clock.t + 23 * 3600                       # within the deadline: still waiting
    orch.intake()
    assert orch.db.sub(sa)["status"] == "WAITING"
    clock.t = Clock.t + 25 * 3600
    orch.intake()
    s = orch.db.sub(sa)
    assert (s["status"], s["reason_code"]) == ("REJECTED", "REVEAL_EXPIRED")
    gh.add_pr(nb + 1, HOTKEYS["A"], "a1", blobs_for("A"))
    gh.prs[nb + 1]["opened_at"] = clock.t
    orch.intake()
    orch.publish_status()
    st = json.loads(pub.get(layout.STATUS))
    assert [r["sid"] for r in st["review"]] == [sb]
    assert next(p for p in st["prs"] if p["pr"] == nb + 1)["reason_code"] == "PARTICIPATED"
    assert orch.db.sub(sa)["status"] == "REJECTED"


def test_the_reveal_deadline_is_github_time_not_our_tick(tmp_path):
    """The deadline is judged on the PR's own opening time: a PR opened after it is too late even if no tick ran in
    between, and one opened in time counts even if first seen after it (an outage of ours costs a miner nothing)."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path, reveal_hours=24)
    orch.bootstrap()
    gh = orch.d.github
    for n, block in (("A", 101), ("B", 102)):
        deposit(chain, n, block)
        commit(chain, n, blobs_for(n), block)
    gh.add_pr(1, HOTKEYS["A"], "a1", blobs_for("A"))
    gh.prs[1]["opened_at"] = Clock.t + 25 * 3600        # after the deadline
    gh.add_pr(2, HOTKEYS["B"], "b1", blobs_for("B"))
    gh.prs[2]["opened_at"] = Clock.t + 23 * 3600        # in time
    clock.t = Clock.t + 30 * 3600                       # both first seen after the deadline
    chain.head = 102
    open_rules(orch, 101)
    a, b = orch.db.sub("0000000101-0001"), orch.db.sub("0000000102-0001")
    assert (a["status"], a["reason_code"]) == ("REJECTED", "REVEAL_EXPIRED")
    assert "after the reveal deadline" in a["reason"]
    assert gh.prs[1]["state"] == "closed" and any(n == 1 and "REVEAL_EXPIRED" in t for n, t in gh.comments)
    assert b["status"] == "QUEUED"


def test_once_revealed_a_later_carrier_is_fine(tmp_path):
    """The deadline is for the first reveal only: once a PR has carried the content it is public, and closing that PR
    and opening another one later (the PR is only the carrier) changes nothing."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path, reveal_hours=24)
    orch.bootstrap()
    gh = orch.d.github
    gh.approve_all = False
    deposit(chain, "A", 101)
    commit(chain, "A", blobs_for("A"), 101)
    gh.add_pr(1, HOTKEYS["A"], "a1", blobs_for("A"))
    gh.prs[1]["opened_at"] = Clock.t + 3600
    clock.t = Clock.t + 3600
    chain.head = 101
    open_rules(orch, 101)
    sid = "0000000101-0001"
    assert orch.db.sub(sid)["status"] == "REVIEW"
    gh.close(1)
    clock.t = Clock.t + 48 * 3600
    orch.intake()
    s = orch.db.sub(sid)
    assert (s["status"], s["reason_code"]) == ("WAITING", "NO_PR") and "void" not in s["reason"]
    gh.add_pr(2, HOTKEYS["A"], "a2", blobs_for("A"))
    gh.prs[2]["opened_at"] = clock.t
    orch.intake()
    assert orch.db.sub(sid)["status"] == "REVIEW" and orch.db.sub(sid)["pr_number"] == 2


def test_no_reveal_hours_no_deadline(tmp_path):
    """Rules without reveal_hours (every version so far) keep today's behaviour: an unrevealed commitment waits."""
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    deposit(chain, "A", 101)
    commit(chain, "A", blobs_for("A"), 101)
    chain.head = 101
    open_rules(orch, 101)
    clock.t = Clock.t + 365 * 24 * 3600
    orch.intake()
    s = orch.db.sub("0000000101-0001")
    assert (s["status"], s["reason_code"]) == ("WAITING", "NO_PR") and "void" not in s["reason"]


def test_an_entry_shows_starting_while_its_machine_sets_up(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    pay(orch, chain, "A", 101, 2)
    chain.head = 101
    open_rules(orch, 101)
    orch.d.fleet.rent((s := orch.db.subs()[0])["sid"], "RTX 4090", 4)
    orch.d.fleet.db.set_machine(s["sid"], state="booting")
    orch.db.update(s["sid"], "QUEUED", reason="machine setup failed")       # left by a machine given up before
    orch.publish_status(every=0)
    (row,) = json.loads(pub.get(layout.STATUS))["queue"]
    assert (row["status"], row["reason"]) == ("STARTING", None)


def test_changing_the_burned_share_reissues_the_weights(tmp_path):
    import time
    orch, chain, clock, pub = make(tmp_path)
    clock.t = int(time.time())                      # weights.json's issued_at is wall time
    orch.d.registry_root = registry_copy(tmp_path, effective_block=100)
    chain.head = 120
    orch.intake()
    orch.bootstrap()
    seq = json.loads(pub.get(layout.WEIGHTS))["sequence"]
    orch.refresh_weights()
    assert json.loads(pub.get(layout.WEIGHTS))["sequence"] == seq          # nothing changed: not re-issued
    orch.d.burn = 0.25
    orch.refresh_weights()
    w = json.loads(pub.get(layout.WEIGHTS))
    assert w["sequence"] == seq + 1 and w["burn"] == 0.25


def test_the_line_is_published_from_the_start(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    orch.bootstrap()
    line = json.loads(pub.get(layout.line(RULES["base"])))
    assert line["champion"] is None and line["entries"] == [] and line["history"] == []


def test_the_review_fee_is_kept_when_review_refuses_and_is_part_of_a_run(tmp_path):
    """fee.review_usd is charged when an entry goes to review. A refused entry keeps it charged (the cost of a bad or
    abusive entry); an entry that runs is charged only what its run used, the fee counting towards it. An account
    covering exactly one run still runs after paying it."""
    orch, chain, clock, pub = make(tmp_path)
    fee = {**RULES["fee"], "assets": ["tao", "alpha"], "review_usd": 20}
    orch.d.registry_root = registry_copy(tmp_path, fee=fee)
    gh = orch.d.github
    gh.approve_all = False
    for n, block in (("A", 101), ("B", 102)):
        pay(orch, chain, n, block, 1)
    chain.head = 102
    open_rules(orch, 100)
    sid_a, sid_b = "0000000101-0001", "0000000102-0001"
    assert orch.db.charged(sid_a) == {-2: 20 * pricing.MUSD} and orch.db.sub(sid_a)["status"] == "REVIEW"
    gh.verdicts[(1, content_digest_of_blobs(blobs_for("A")))] = "changes"
    gh.verdicts[(2, content_digest_of_blobs(blobs_for("B")))] = "approved"
    orch.intake()
    assert orch.db.sub(sid_a)["reason_code"] == "REVIEW_CHANGES" and orch.db.charged(sid_a) == {-2: 20 * pricing.MUSD}
    orch.publish_status(every=0)
    assert json.loads(pub.get(layout.STATUS))["accounts"][COLDKEYS["A"]]["charged_usd"] == 20   # public
    s = orch.db.sub(sid_b)
    assert s["status"] == "QUEUED"                       # B paid the fee and its account (one run) still covers the run
    assert orch.db.accounts()[COLDKEYS["B"]]["held_usd"] == RUN_USD - 20
    orch.schedule()
    kick = clock.t
    clock.t = kick + 3600 + 600
    orch.d.fleet.done.add(sid_b)
    orch.supervise()
    used = pricing.used_musd(orch.rules_for(sid_b), clock.t - kick)
    assert sum(orch.db.charged(sid_b).values()) == used                  # the run's cost; the fee went back
    assert orch.db.accounts()[COLDKEYS["B"]]["held_usd"] == 0


def test_a_hotkey_moved_under_another_coldkey_cannot_spend_it(tmp_path):
    """A coldkey swap needs no consent from its destination. A hotkey moved under coldkey B after its registration
    cannot enter (it would spend B's account); a hotkey moved after its commitment is still paid by the coldkey that
    owned it then."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path)
    moved = {}                                            # hotkey -> (from block, new coldkey)
    plain = chain.owner_of
    chain.owner_of = lambda hk, block: (moved[hk][1] if hk in moved and block >= moved[hk][0] else plain(hk, block))
    deposit(chain, "B", 100, 2 * FEE)                     # B has money; A has none
    moved[HOTKEYS["A"]] = (50, COLDKEYS["B"])             # A's hotkey moved under B before A commits
    pay(orch, chain, "C", 103, 1)                         # C pays for itself, then its hotkey moves under B
    commit(chain, "A", blobs_for("A"), 101)
    orch.d.github.add_pr(9, HOTKEYS["A"], "a9", blobs_for("A"))
    moved[HOTKEYS["C"]] = (104, COLDKEYS["B"])
    chain.head = 110
    open_rules(orch, 100)
    a, c = orch.db.of_hotkey(HOTKEYS["A"]), orch.db.of_hotkey(HOTKEYS["C"])
    assert (a["status"], a["reason_code"]) == ("REJECTED", "HOTKEY_MOVED")
    assert c["coldkey"] == COLDKEYS["C"] and c["status"] == "QUEUED"
    assert orch.db.accounts()[COLDKEYS["B"]]["held_usd"] == 0 and orch.db.accounts()[COLDKEYS["B"]]["charged_usd"] == 0


def test_a_failed_chain_read_never_skips_the_ownership_check(tmp_path):
    """The registration's owner is read before the entry is recorded: a failed read retries the block (the entry is
    then still checked); a hotkey with no registration left at its commitment's block is not checked."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.registry_root = registry_copy(tmp_path)
    real, calls = chain.registration_owner, []

    def flaky(hk, netuid, block):
        calls.append(hk)
        if len(calls) == 1:
            raise ConnectionError("archive node unreachable")
        if hk == HOTKEYS["B"]:
            return None, None                              # deregistered later in its commitment's block
        return 1, COLDKEYS["D"]                            # A's hotkey belonged to D at its registration
    chain.registration_owner = flaky
    commit(chain, "A", blobs_for("A"), 101)
    commit(chain, "B", blobs_for("B"), 102)
    chain.head = 102
    orch.d.registry_root = registry_copy(tmp_path, effective_block=100)
    with pytest.raises(ConnectionError):
        orch.intake()
    assert orch.db.subs() == []                            # nothing recorded on a failed read
    orch.intake()
    a, b = orch.db.of_hotkey(HOTKEYS["A"]), orch.db.of_hotkey(HOTKEYS["B"])
    assert (a["status"], a["reason_code"]) == ("REJECTED", "HOTKEY_MOVED")
    assert b["reason_code"] != "HOTKEY_MOVED"             # no registration to compare with: not checked
    chain.registration_owner = real



def test_a_waiting_entry_never_stalls_a_machine_already_rented(tmp_path):
    """Machines are rented in priority order, but an earlier entry that cannot get one (here: backing off, then the
    only slot is taken) must not stop a later entry's machine from being set up and its run from starting."""
    orch, chain, clock, pub = make(tmp_path)
    orch.d.max_concurrent = 1
    for n, block in (("A", 101), ("B", 102)):
        pay(orch, chain, n, block, 1)
    chain.head = 102
    open_rules(orch, 101)
    sa, sb = "0000000101-0001", "0000000102-0001"
    orch.db.set(f"backoff:{sa}:until", clock.t + 600)  # A's rent failed a moment ago
    plain = orch.d.fleet.rent

    def rent_booting(purpose, gpu, gpus):
        plain(purpose, gpu, gpus)
        orch.d.fleet.db.set_machine(purpose, state="booting")
    orch.d.fleet.rent = rent_booting
    orch.schedule()
    assert orch.d.fleet.get(sb)["state"] == "booting" and orch.db.sub(sb)["status"] == "QUEUED"
    clock.t += 601                                      # A is due again; the only slot is B's
    orch.d.fleet.db.set_machine(sb, state="ready")
    orch.schedule()
    assert orch.db.sub(sb)["status"] == "RUNNING" and orch.d.fleet.get(sa) is None


def test_a_champions_pr_is_merged_even_if_github_fails_once(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    pay(orch, chain, "A", 101, 1)
    chain.head = 101
    open_rules(orch, 101)
    plain, failed = gh.comment, []

    def flaky(number, text):
        if not failed:
            failed.append(number)
            raise ConnectionError("GitHub hiccup")
        plain(number, text)
    gh.comment = flaky
    run_until_quiet(orch, clock)
    sid = "0000000101-0001"
    assert orch.db.champion(BASE)["sid"] == sid and failed == [1]
    orch.settle_prs()
    assert gh.merged == [1] and gh.prs[1]["state"] == "closed" and orch.db.sub(sid)["record"]["pr_result"]["closed"]



def test_index_follows_a_new_rules_version_of_the_same_base(tmp_path):
    """A later rules version of the line in force (e.g. a review fee from block 130) is what miners read from
    index.json as soon as it takes effect, not only when the weights are next issued."""
    from axonry.common.canon import write_json
    orch, chain, clock, pub = make(tmp_path)
    root = registry_copy(tmp_path, effective_block=100)
    orch.d.registry_root = root
    rules = json.loads(next((root / "rules" / BASE).glob("*.json")).read_text())
    write_json(root / "rules" / BASE / "v2.json", {**rules, "effective_block": 130,       # deployed, staged
                                                   "fee": {**rules["fee"], "review_usd": 20}})
    chain.head = 120
    orch.intake()
    orch.bootstrap()
    v1 = json.loads(pub.get(layout.INDEX))["rules"]
    orch.refresh_weights()
    assert json.loads(pub.get(layout.INDEX))["rules"] == v1             # staged, not in force yet
    chain.head = 131
    orch.refresh_weights()
    idx = json.loads(pub.get(layout.INDEX))
    assert idx["rules"] != v1 and json.loads(pub.get(idx["rules"]))["fee"]["review_usd"] == 20


def admitted(orch, n):
    """n candidates ADMITTED directly (their runs and admission are other tests' business)."""
    rules = registry.load_rules_versions(BASE, orch.d.registry_root)[-1]
    sids = [f"00000002{i:02d}-0001" for i in range(n)]
    for i, sid in enumerate(sids):
        orch.db.add_submission(sid, BASE, f"hk{i}", f"ck{i}", "ADMITTED",
                               {"rules_sha256": rules.sha256, "base": BASE, "timeline": {}}, f"d{i}", (200 + i, 1))
        orch.d.executor.specs[sid] = next(iter(PLANS.values()))["spec"]
    return sids


def test_a_long_evaluation_queue_takes_a_second_machine_until_it_is_short_again(tmp_path):
    from axonry.pool.machine import EVAL
    orch, chain, clock, pub = make(tmp_path)
    orch.d.eval_machines, orch.d.eval_scale_up, orch.d.eval_scale_down = 2, 2, 2
    s1, s2, s3 = admitted(orch, 3)
    ex, fleet = orch.d.executor, orch.d.fleet
    orch.eval_machine()                                  # 3 waiting > 2: both evaluation machines
    assert fleet.get(EVAL) and fleet.get("eval-2")
    orch.evaluate()                                      # one model per machine: the base and the first candidate
    assert ex.evaluated == [f"base:{BASE}", s1] and {j["machine"] for j in orch._eval_jobs().values()} == {EVAL, "eval-2"}
    orch.evaluate()                                      # both finished; the next two start side by side
    assert orch.db.sub(s1)["status"] == "SCORED" and ex.evaluated[2:] == [s2, s3]
    orch.evaluate()
    assert orch.db.sub(s3)["status"] == "SCORED" and not orch._eval_jobs()
    orch.eval_machine()                                  # nothing waits: the second one goes, the first idles
    assert "eval-2" in fleet.released and fleet.get(EVAL)


def test_an_evaluation_whose_machine_is_gone_starts_again_on_another(tmp_path):
    from axonry.pool.machine import EVAL
    orch, chain, clock, pub = make(tmp_path)
    orch.d.eval_machines, orch.d.eval_scale_up = 2, 0
    s1, = admitted(orch, 1)
    orch.eval_machine()
    orch.evaluate()                                      # the base on the first machine, the candidate on the second
    assert orch._eval_jobs()[s1]["machine"] == "eval-2"
    orch.d.fleet.release("eval-2", "vanished")
    for job in [j for j in orch.d.executor.jobs if j.startswith("eval-2/")]:
        orch.d.executor.jobs.pop(job)                    # its job is gone with it
    orch.evaluate()                                      # dropped; the first machine (done with the base) takes it
    assert orch._eval_jobs()[s1]["machine"] == EVAL


def test_after_a_review_asks_for_changes_only_deletions_are_reviewed_again(tmp_path):
    """The content can only shrink: a PR with the revealed content minus lines or files carries the entry and is
    reviewed anew (keeping its priority and the one review fee); anything added or rewritten carries nothing."""
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    full = blobs_for("A") | {"METHOD.md": b"method of A\nscores on the benchmarks\nmore of A\n",
                             "data/words.txt": b"alpha\nbeta\n"}
    n = pay(orch, chain, "A", 101, 1, blobs=full)
    chain.head = 101
    open_rules(orch, 101)
    sid = "0000000101-0001"
    gh.verdicts[(n, content_digest_of_blobs(full))] = "changes"
    orch.intake()
    assert orch.db.sub(sid)["reason_code"] == "REVIEW_CHANGES"
    added = full | {"METHOD.md": b"method of A\nmore of A\nsomething new\n"}
    gh.push(n, "a-added", added)
    orch.intake()
    assert orch.db.sub(sid)["reason_code"] == "NO_PR"                     # not a reduction: carries nothing
    reduced = {k: v for k, v in full.items() if k != "data/words.txt"} | {"METHOD.md": b"method of A\nmore of A\n"}
    gh.push(n, "a-reduced", reduced)
    orch.intake()
    s = orch.db.sub(sid)
    assert s["status"] == "REVIEW" and s["record"]["content"] == content_digest_of_blobs(reduced)
    gh.verdicts[(n, content_digest_of_blobs(reduced))] = "approved"
    orch.intake()
    orch.schedule()
    s = orch.db.sub(sid)
    assert s["status"] == "RUNNING" and s["locked_commit"] == "a-reduced"
    assert orch.d.runner.methods[sid] == "method of A\nmore of A\n"


def test_a_binary_file_is_refused_whatever_its_name(tmp_path):
    from axonry.intake.github import check_files
    blobs = blobs_for("A") | {"data/dirs.txt": b"\x93NUMPY\x01\x00v\x00{'descr'"}
    assert "not readable text" in check_files(blobs, COMP_RULES)
    assert check_files(blobs_for("A") | {"data/words.txt": "ünïcode words\n".encode()}, COMP_RULES) is None


def test_content_that_breaks_the_rules_waits_until_deleting_fixes_it(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    full = blobs_for("A") | {"data/dirs.txt": b"\x93NUMPY\x01\x00binary"}
    n = pay(orch, chain, "A", 101, 1, blobs=full)
    chain.head = 101
    open_rules(orch, 101)
    s = orch.db.sub("0000000101-0001")
    assert s["status"] == "WAITING" and "not readable text" in s["reason"] and "Only deletions" in s["reason"]
    gh.push(n, "a-fixed", blobs_for("A"))
    orch.intake()
    assert orch.db.sub("0000000101-0001")["status"] == "REVIEW"


def refused_entry(tmp_path, **rules_patch):
    """A's entry, its content refused by review: (orch, chain, clock, pr number, the refused content)."""
    orch, chain, clock, pub = make(tmp_path)
    if rules_patch:
        orch.d.registry_root = registry_copy(tmp_path, **rules_patch)
    gh = orch.d.github
    gh.approve_all = False
    full = blobs_for("A") | {"METHOD.md": b"method of A\nscores on the benchmarks\nmore of A\n"}
    n = pay(orch, chain, "A", 101, 1, blobs=full)
    chain.head = 101
    open_rules(orch, 101)
    gh.verdicts[(n, content_digest_of_blobs(full))] = "changes"
    orch.intake()
    assert orch.db.sub("0000000101-0001")["reason_code"] == "REVIEW_CHANGES"
    return orch, chain, clock, n, full


def test_a_refused_content_stays_refused_on_any_pull_request(tmp_path):
    orch, chain, clock, n, full = refused_entry(tmp_path)
    gh = orch.d.github
    gh.prs[n]["state"] = "closed"
    gh.add_pr(n + 1, HOTKEYS["A"], "a-again", full)        # the same content again, on a new PR: no second review
    orch.intake()
    s = orch.db.sub("0000000101-0001")
    assert (s["status"], s["reason_code"]) == ("WAITING", "REVIEW_CHANGES") and s["pr_number"] == n + 1


def test_the_content_only_shrinks_and_a_fix_is_due_in_time(tmp_path):
    orch, chain, clock, n, full = refused_entry(tmp_path, reveal_hours=24)
    gh, sid = orch.d.github, "0000000101-0001"
    reduced = full | {"METHOD.md": b"method of A\nmore of A\n"}
    gh.push(n, "a-reduced", reduced)
    orch.intake()
    assert orch.db.sub(sid)["status"] == "REVIEW"
    gh.push(n, "a-full-again", full)                     # an earlier, larger version never carries it again
    orch.intake()
    assert orch.db.sub(sid)["reason_code"] == "NO_PR"
    gh.push(n, "a-reduced-2", reduced)
    orch.intake()
    gh.verdicts[(n, content_digest_of_blobs(reduced))] = "changes"
    orch.intake()
    assert orch.db.sub(sid)["reason_code"] == "REVIEW_CHANGES"
    clock.t += 25 * 3600                                 # nothing fixed it within reveal_hours: the entry ends
    orch.intake()
    s = orch.db.sub(sid)
    assert (s["status"], s["reason_code"]) == ("REJECTED", "REVIEW_CHANGES") and "nothing fixed it" in s["reason"]


def test_a_reduction_counts_only_when_a_fix_is_due(tmp_path):
    orch, chain, clock, pub = make(tmp_path)
    gh = orch.d.github
    gh.approve_all = False
    full = blobs_for("A") | {"METHOD.md": b"method of A\nmore of A\n"}
    n = pay(orch, chain, "A", 101, 1, blobs=full)
    chain.head = 101
    open_rules(orch, 101)
    gh.push(n, "a-reduced", full | {"METHOD.md": b"method of A\n"})
    orch.intake()
    assert orch.db.sub("0000000101-0001")["reason_code"] == "NO_PR"      # nothing asked for a fix


def test_an_entry_carried_before_reductions_keeps_its_first_content_for_review(tmp_path):
    orch, chain, clock, n, full = refused_entry(tmp_path)
    sid = "0000000101-0001"
    rec = orch.db.sub(sid)["record"]
    orch.db.c.execute("UPDATE submissions SET record=? WHERE sid=?",          # as recorded before reductions existed
                      (json.dumps({k: v for k, v in rec.items() if k not in ("revealed", "content")}), sid))
    first = orch.db.sub(sid)["head_sha"]
    orch.d.github.push(n, "a-reduced", full | {"METHOD.md": b"method of A\nmore of A\n"})
    orch.intake()
    s = orch.db.sub(sid)
    assert s["status"] == "REVIEW" and s["record"]["revealed"]["commit"] == first
    orch.publish_status(every=0)
    item = json.loads(orch.d.public.get(layout.STATUS))["review"][0]
    assert item["revealed"]["commit"] == first and item["commit"] == "a-reduced"


def test_fixing_has_one_window_from_the_first_problem(tmp_path):
    orch, chain, clock, n, full = refused_entry(tmp_path, reveal_hours=24)
    gh, sid = orch.d.github, "0000000101-0001"
    clock.t += 20 * 3600
    reduced = full | {"METHOD.md": b"method of A\nmore of A\n"}
    gh.push(n, "a-reduced", reduced)
    orch.intake()
    clock.t += 3 * 3600 + 1800                           # past the window while the reduction waits for review
    orch.intake()
    assert orch.db.sub(sid)["status"] == "REVIEW"        # the review's wait is never the miner's
    gh.verdicts[(n, content_digest_of_blobs(reduced))] = "changes"
    clock.t += 3600
    orch.intake()                                        # refused again after the window: no second one
    s = orch.db.sub(sid)
    assert (s["status"], s["reason_code"]) == ("REJECTED", "REVIEW_CHANGES")
