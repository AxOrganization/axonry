"""axonry-orchestrator: the owner's service and its maintenance commands.

  axonry-orchestrator run [--role main|eval|all] [--once]
                                                     main: intake, schedule (rents run machines), supervise, decide,
                                                     weights, janitor (bootstraps public files first); eval: the
                                                     evaluation machines (taken on demand), collect, evaluate. Run one
                                                     process per role (shared SQLite, WAL).
  axonry-orchestrator status                            submissions by status
  axonry-orchestrator refresh-agents                    keep the agent backend's sign-ins valid while nothing runs
                                                     (a timer, every 10 min; `run --role main` does it every tick)
  axonry-orchestrator disqualify SID --reason TEXT      public owner ruling: an evaluated submission broke the rules
  axonry-orchestrator calibrate [--limit N --samples K]  A/A noise per benchmark on the evaluation machine
  axonry-orchestrator restore-state [--key K]           replace the local state DB by a snapshot from the private bucket
  axonry-orchestrator registry validate                 cross-check bases / benchmarks / rules versions
  axonry-orchestrator registry pin-base BASE_ID --model-dir DIR
                                                     write registry/bases/<id>.json from the competition repo's profile
                                                     (hf repo + revision) and the sha256 of every file in DIR
  axonry-orchestrator registry pin-bench KEY            items_manifest_sha256 of the items built now (benchmarks/sources.py)
  axonry-orchestrator registry pin-competition          competition commit + rules.json sha256 into the latest rules
  axonry-orchestrator publish-registry                  publish definitions to the public bucket
  axonry-orchestrator publish-items                     build the rules' benchmark items (checked against their frozen
                                                     manifests) and publish them beside their definitions
Configuration: config/axonry.toml (see config/example.toml); secrets from the environment.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from .. import pricing, registry
from ..common.canon import read_json, sha256_file, sha256_hex, write_json
from ..common.config import Config, private_store, public_store


def fleet(cfg, db, rules):
    """The GPU machines ([machines]): the owned hosts ([[machines.owned]]) first, then vast.ai ([vast], if enabled)."""
    from ..evaluation import benchdata  # noqa: PLC0415
    from ..pool import Machines  # noqa: PLC0415
    from ..pool.machine import setup_bundle  # noqa: PLC0415
    from ..pool.owned import OwnedPool  # noqa: PLC0415
    from ..pool.vast import Fleet, VastAPI  # noqa: PLC0415
    root = Path(__file__).resolve().parents[3]
    memo = {}

    benches = [registry.load_benchmark(k).doc for k in rules.benchmark_keys()]

    def bundle():   # built once per process, from this checkout and the pinned competition repo
        if "path" not in memo:
            memo["path"] = setup_bundle(root, cfg.path_of("competition.local_path"), [registry.load_base(rules["base"])],
                                        benches, cfg.path_of("state.work_dir") / "setup.tgz")
        return memo["path"]

    def bundle_url(path):   # a vast.ai machine downloads the bundle from the private bucket (a presigned URL)
        priv = private_store(cfg)
        if getattr(priv, "s3", None) is None:
            return None
        key = f"setup/{sha256_file(path)}.tgz"
        if priv.last_modified(key) is None:
            priv.put_file(key, path)
        return priv.s3.generate_presigned_url("get_object", Params={"Bucket": priv.bucket, "Key": key}, ExpiresIn=6 * 3600)

    def items():    # an evaluation machine's benchmark items, built once per set of benchmarks
        return benchdata.build(cfg.path_of("competition.local_path"), benches, cfg.path_of("state.work_dir"))
    key, image, proxy = cfg.require("machines.ssh_key"), cfg.require("machines.image"), cfg.get("machines.proxy_command")
    state = cfg.path_of("state.work_dir") / "machines"
    hosts = cfg.get("machines.owned") or []
    owned = OwnedPool(hosts, db, image=image, ssh_key=key, bundle=bundle, items=items, state_dir=state,
                      proxy_command=proxy) if hosts else None
    vast = Fleet(VastAPI(cfg.secret("vast.api_key")), db, image=image, ssh_key=key, bundle=bundle, state_dir=state,
                 disk_gb=int(cfg.get("vast.disk_gb", 150)), min_reliability=float(cfg.get("vast.min_reliability", 0.95)),
                 min_down_mbps=int(cfg.get("vast.min_down_mbps", 200)),
                 exclude_countries=cfg.get("vast.exclude_countries", ["CN"]),
                 verified_only=bool(cfg.get("vast.verified_only", False)), min_cuda=float(cfg.get("vast.min_cuda", 0)),
                 setup_minutes=int(cfg.get("vast.setup_minutes", 45)),
                 proxy_command=proxy, bundle_url=bundle_url, items=items) if cfg.get("vast.enabled", False) else None
    return Machines(db, str(Path(key).expanduser()), owned, vast, items)


def eval_executor(cfg, fl):
    """Every evaluation runs on all GPUs of an evaluation machine."""
    from ..evaluation.executor import ExecError, SshExecutor  # noqa: PLC0415

    def machine(purpose):
        row = fl.get(purpose)
        if row is None or row["state"] != "ready":
            raise ExecError(f"evaluation machine {purpose} not ready")
        return fl.machine(row)
    return SshExecutor(machine, fl.key, int(cfg.get("machines.eval_gpus", 8)),
                       concurrency_per_gpu=int(cfg.get("eval.concurrency_per_gpu", 256)))


def cached_price(ttl=300, retry=60, lookup=pricing.tao_usd_at, clock=time.time):
    """TAO/USD at a moment, for crediting deposits: one lookup per `ttl`-second bucket of that moment (deposits close
    together share it); a failed lookup is retried after `retry` seconds of wall clock, so the sources are not
    hammered while they are down and a deposit is priced as soon as they are back."""
    memo = {}

    def price_at(ts):
        key = int(ts) // ttl
        hit = memo.get(key)
        if hit is None or ("error" in hit and clock() - hit["at"] >= retry):
            try:
                hit = {"value": lookup(ts)}
            except pricing.PriceError as e:
                hit = {"error": e, "at": clock()}
            memo[key] = hit
        if "error" in hit:
            raise hit["error"]
        return hit["value"]
    return price_at


def build(cfg: Config):
    """Wire the production dependencies from the configuration."""
    from .. import creds  # noqa: PLC0415
    from ..agent.factory import make_runner  # noqa: PLC0415
    from ..chain.subtensor import Chain  # noqa: PLC0415
    from ..common.sign import load_keypair  # noqa: PLC0415
    from ..intake.github import GitHub, scan  # noqa: PLC0415
    from .db import DB  # noqa: PLC0415
    from .service import Deps, Orchestrator  # noqa: PLC0415

    rules = registry.latest_rules()
    comp_rules = competition_rules(cfg.path_of("competition.local_path"), rules)
    gh = GitHub(rules["competition"]["repo"], cfg.secret("competition.github_token"))
    owner = load_keypair(cfg.require("owner.wallet"), cfg.require("owner.hotkey"), cfg.get("owner.path"))
    db = DB(cfg.path_of("state.db"))
    fl, runner = fleet(cfg, db, rules), make_runner(cfg)

    def issue(store, prefix, ttl):
        """Temporary credentials for one prefix of the store the orchestrator itself uses: what a machine writes
        there is exactly what the orchestrator then reads or publishes."""
        if getattr(store, "bucket", None) is None:
            raise RuntimeError("temporary credentials need an R2 store (storage.backend = r2)")
        return creds.issue(cfg.require("r2.account_id"), cfg.secret("r2.cf_api_token"),
                           cfg.secret("r2.temp_parent_access_key_id"), store.bucket, prefix, ttl,
                           cfg.require("r2.endpoint"))
    return Orchestrator(Deps(
        db=db, chain=Chain(cfg.get("chain.network", "finney")), github=gh, fleet=fl, runner=runner,
        executor=eval_executor(cfg, fl), public=public_store(cfg), private=private_store(cfg),
        owner_keypair=check_published(cfg, rules, owner), comp_rules=comp_rules, price_at=cached_price(),
        scan_prs=lambda: scan(gh, comp_rules), issue_creds=issue, work_dir=cfg.path_of("state.work_dir"),
        max_concurrent=min(runner.capacity, int(cfg.get("machines.max_runs", runner.capacity))),
        eval_gpu_name=cfg.get("machines.eval_gpu_name", "RTX 4090"), eval_gpus=int(cfg.get("machines.eval_gpus", 8)),
        eval_idle_minutes=int(cfg.get("machines.eval_idle_minutes", 20)), burn=owner_burn(cfg),
        eval_machines=int(cfg.get("machines.eval_machines", 1)),
        eval_scale_up=int(cfg.get("machines.eval_scale_up", 10)),
        eval_scale_down=int(cfg.get("machines.eval_scale_down", 5)),
        eval_image=cfg.get("machines.image")))


def owner_burn(cfg) -> float:
    burn = float(cfg.get("owner.burn", 0.0))
    if not 0 <= burn <= 1:
        raise SystemExit(f"owner.burn must be between 0 and 1, got {burn}")
    return burn


def load_dotenv(path=".env"):
    """KEY=VALUE lines into the environment (existing variables win); secrets stay out of the TOML config."""
    import os  # noqa: PLC0415
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def check_published(cfg, rules, keypair):
    """The constants validators and miners ship with (axonry_validator.network) must describe this deployment."""
    from axonry_validator.network import NETWORKS  # noqa: PLC0415
    net = NETWORKS.get(cfg.get("chain.network", "finney"), {})
    if net.get("netuid") is not None and net["netuid"] != rules["netuid"]:
        raise SystemExit(f"rules netuid {rules['netuid']} != published netuid {net['netuid']}")
    if net.get("owner") is not None and net["owner"] != keypair.ss58_address:
        raise SystemExit(f"owner key {keypair.ss58_address} is not the published owner {net['owner']}")
    return keypair


def competition_rules(repo: Path, rules) -> dict:
    """The competition protocol the rules pin. Refuses to run on a checkout that does not contain the pinned commit or
    whose rules.json differs from the pin, so the protocol applied is always the one the rules publish (later toolkit
    commits that keep rules.json apply without a new rules version)."""
    pin = rules["competition"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    raw = (Path(repo) / "rules.json").read_bytes()
    contains = subprocess.run(["git", "merge-base", "--is-ancestor", pin["commit"], "HEAD"], cwd=repo,
                              capture_output=True).returncode == 0
    if not contains or sha256_hex(raw) != pin["rules_sha256"]:
        raise SystemExit(f"competition checkout {repo} is at {head[:12]} (rules {sha256_hex(raw)[:12]}), but the rules "
                         f"pin {pin['commit'][:12]} (rules {pin['rules_sha256'][:12]}): "
                         "git submodule update, or pin a new version (registry pin-competition)")
    return json.loads(raw)


def cmd_run(cfg, a):
    orch = build(cfg)
    if a.role in ("main", "all") and orch.db.get("cursor") is None:
        try:
            orch.intake()         # the first rules' effective_block: nothing earlier is read
        except ValueError as e:
            raise SystemExit(str(e))
    if a.role in ("main", "all"):
        orch.bootstrap()
    interval = int(cfg.get("state.tick_seconds", 60))
    while True:
        orch.tick(a.role)
        if a.once:
            return
        time.sleep(interval)


def cmd_restore(cfg, a):
    """Put a state snapshot from the private bucket in place of the local DB (the orchestrator must be stopped)."""
    from ..common import layout  # noqa: PLC0415
    store = private_store(cfg)
    keys = sorted(k for k, _ in store.list(layout.STATE_PREFIX))
    if not keys:
        raise SystemExit("no snapshots")
    key = a.key or keys[-1]
    db = cfg.path_of("state.db")
    if db.exists():
        db.rename(db.with_suffix(f".before-restore-{int(time.time())}"))
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(store.get(key))
    print(f"restored {key} -> {db}")


def cmd_calibrate(cfg, a):
    from ..evaluation.calibrate import calibrate, scaled_registry  # noqa: PLC0415
    rules = registry.latest_rules()
    root = registry.ROOT
    if a.limit or a.samples:   # a scaled copy of the registry, never committed
        root = scaled_registry(Path(".state/calibrate/registry"), limit=a.limit, samples=a.samples)
        rules = registry.load_rules_versions(rules["base"], root)[-1]
    from ..pool.machine import EVAL  # noqa: PLC0415
    from .db import DB  # noqa: PLC0415
    fl = fleet(cfg, DB(cfg.path_of("state.db")), rules)
    rented = fl.get(EVAL) is None
    if rented:
        fl.rent(EVAL, cfg.get("machines.eval_gpu_name", "RTX 4090"), int(cfg.get("machines.eval_gpus", 8)))
    try:
        while (state := fl.advance(EVAL)) not in ("ready", "failed"):
            print(f"evaluation machine: {state}", flush=True)
            time.sleep(30)
        if state == "failed":
            raise SystemExit("the evaluation machine failed to set up (see the log); run calibrate again")
        calibrate(eval_executor(cfg, fl), EVAL, rules, root, out=Path(".state/calibration.json"))
    finally:
        if rented:
            fl.release(EVAL)


def cmd_status(cfg, _a):
    from .db import DB  # noqa: PLC0415
    db = DB(cfg.path_of("state.db"))
    for s in db.subs():
        print(f"{s['sid']}  {s['hotkey'][:12]}…  {s['status']:<10} {s['reason_code'] or ''}")


def pin_base(base_id, model_dir, comp_repo, root=registry.ROOT):
    """Base definition generated from the competition repo's profile (single source of hf repo + revision)."""
    out = subprocess.run([sys.executable, "-c", "import json, wsp.profiles as p; print(json.dumps(p.get(%r)))" % base_id],
                         cwd=comp_repo, capture_output=True, text=True, check=True)
    prof = json.loads(out.stdout)
    model_dir = Path(model_dir)
    files = {p.relative_to(model_dir).as_posix(): sha256_file(p) for p in sorted(model_dir.rglob("*"))
             if p.is_file() and p.name != "VERIFIED"}
    doc = {"schema": "axonry.base/1", "id": base_id, "hf_repo": prof["hf"], "revision": prof["revision"], "files": files,
           "about": "Generated by `axonry-orchestrator registry pin-base` from the competition repo's wsp/profiles.py; "
                    "files = sha256 of every file of the pinned snapshot."}
    write_json(Path(root) / "bases" / f"{base_id}.json", doc)
    return doc


def pin_bench(key, comp_repo, root=registry.ROOT):
    from ..evaluation import benchdata  # noqa: PLC0415
    b = registry.load_benchmark(key, root)
    sha = benchdata.manifests(Path(comp_repo), [b.doc["task"]])[b.doc["task"]]
    doc = {**b.doc, "items_manifest_sha256": sha}
    write_json(Path(root) / "benchmarks" / f"{key}.json", doc)
    return doc


def pin_competition(comp_repo, root=registry.ROOT):
    """Record the competition repo commit and rules.json sha256 in the latest rules version (edit it only before it
    takes effect; afterwards add a new version instead)."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=comp_repo, capture_output=True, text=True,
                            check=True).stdout.strip()
    rules_sha = sha256_hex(Path(comp_repo, "rules.json").read_bytes())
    path = sorted((Path(root) / "rules" / registry.latest_rules(root)["base"]).glob("*.json"))[-1]
    doc = read_json(path)
    doc["competition"] = {**doc["competition"], "commit": commit, "rules_sha256": rules_sha}
    write_json(path, doc)
    return path, commit, rules_sha


def main(argv=None):
    ap = argparse.ArgumentParser(prog="axonry-orchestrator", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--role", choices=("main", "eval", "all"), default="all")
    r.add_argument("--once", action="store_true")
    sub.add_parser("status")
    sub.add_parser("refresh-agents")
    ca = sub.add_parser("calibrate", help="A/A noise of every benchmark on the evaluation GPUs (the base model against itself)")
    ca.add_argument("--limit", type=int, help="items per benchmark (default: the definitions')")
    ca.add_argument("--samples", type=int, help="samples per item (default: the definitions')")
    rs = sub.add_parser("restore-state", help="replace the local state DB by a snapshot (default: the latest)")
    rs.add_argument("--key")
    sub.add_parser("publish-registry")
    sub.add_parser("publish-items")
    dq = sub.add_parser("disqualify", help="public owner ruling: an evaluated submission broke the rules")
    dq.add_argument("sid")
    dq.add_argument("--reason", required=True)
    reg = sub.add_parser("registry").add_subparsers(dest="reg", required=True)
    reg.add_parser("validate")
    pb = reg.add_parser("pin-base")
    pb.add_argument("base_id")
    pb.add_argument("--model-dir", required=True)
    reg.add_parser("pin-bench").add_argument("key")
    reg.add_parser("pin-competition")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if a.cmd == "registry" and a.reg == "validate":
        errs = registry.validate()
        print("\n".join(errs) or "registry ok")
        return 1 if errs else 0
    load_dotenv()
    cfg = Config.load(a.config)
    if a.cmd == "run":
        cmd_run(cfg, a)
    elif a.cmd == "status":
        cmd_status(cfg, a)
    elif a.cmd == "refresh-agents":
        from ..agent.factory import make_runner  # noqa: PLC0415
        make_runner(cfg).keep_alive()
    elif a.cmd == "restore-state":
        cmd_restore(cfg, a)
    elif a.cmd == "calibrate":
        cmd_calibrate(cfg, a)
    elif a.cmd == "disqualify":
        build(cfg).disqualify(a.sid, a.reason)
    elif a.cmd == "publish-registry":
        from ..publish.publisher import publish_registry  # noqa: PLC0415
        publish_registry(public_store(cfg))
    elif a.cmd == "publish-items":
        from ..evaluation import benchdata  # noqa: PLC0415
        rules = registry.latest_rules()
        docs = [registry.load_benchmark(k).doc for k in rules.benchmark_keys()]
        pack = benchdata.build(cfg.path_of("competition.local_path"), docs, cfg.path_of("state.work_dir"))
        for key in benchdata.publish(public_store(cfg), pack, docs):
            print(key)
    elif a.reg == "pin-base":
        print(json.dumps(pin_base(a.base_id, a.model_dir, cfg.path_of("competition.local_path")), indent=1)[:2000])
    elif a.reg == "pin-bench":
        print(pin_bench(a.key, cfg.path_of("competition.local_path"))["items_manifest_sha256"])
    elif a.reg == "pin-competition":
        print(pin_competition(cfg.path_of("competition.local_path")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
