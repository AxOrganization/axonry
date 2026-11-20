"""Registry of definitions: bases, benchmarks and rules versions.

Source files live in the repo under registry/ and are published unchanged to the public bucket:
  registry/bases/<base_id>.json                -> bases/<base_id>.json
  registry/benchmarks/<id>@<version>.json      -> benchmarks/<id>@<version>.json
  registry/rules/<base_id>/<name>.json         -> rules/<sha256>.json
The competition runs on one base model at a time. A rules version takes effect at `effective_block` (the latest one
at or before a block is in force); the versions of one base form its line, whose champion carries on across them; a
new base starts its own line. A submission keeps the version in force when it entered the queue. Nothing else in the
code names a concrete base or benchmark.
"""
from dataclasses import dataclass
from pathlib import Path

from .common.canon import canonical, read_json, sha256_hex

ROOT = Path(__file__).resolve().parents[2] / "registry"

BASE_FIELDS = {"schema", "id", "hf_repo", "revision", "files"}
BENCH_FIELDS = {"schema", "id", "version", "adapter", "task", "samples_per_item", "items_manifest_sha256", "judge"}
RULES_FIELDS = {"schema", "effective_block", "netuid", "base", "benchmarks", "judge", "fee", "run",
                "competition", "review"}


class RegistryError(ValueError):
    pass


def _need(doc, fields, what):
    missing = fields - set(doc)
    if missing:
        raise RegistryError(f"{what}: missing {sorted(missing)}")


@dataclass(frozen=True)
class Benchmark:
    id: str
    version: int
    doc: dict

    @property
    def key(self):
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class Rules:
    doc: dict

    @property
    def sha256(self):
        return sha256_hex(canonical(self.doc))

    def benchmark_keys(self):
        return [b["key"] for b in self.doc["benchmarks"]]

    def judge_params(self, bench_key):
        """Per-benchmark thresholds: rules override > benchmark default."""
        entry = next(b for b in self.doc["benchmarks"] if b["key"] == bench_key)
        return {**load_benchmark(bench_key).doc["judge"], **entry.get("judge", {})}

    def __getitem__(self, k):
        return self.doc[k]


def load_base(base_id, root=ROOT):
    p = Path(root) / "bases" / f"{base_id}.json"
    if not p.exists():
        raise RegistryError(f"unknown base {base_id}")
    doc = read_json(p)
    _need(doc, BASE_FIELDS, f"base {base_id}")
    if doc["id"] != base_id:
        raise RegistryError(f"base file {p.name} has id {doc['id']}")
    return doc


def base_name(doc) -> str:
    """A pinned base's full name, as an edit names it when it starts from the base: <hf_repo>@<revision>."""
    return f"{doc['hf_repo']}@{doc['revision']}"


def base_dirname(doc) -> str:
    """Directory name of a pinned base, as wsp expects under WSP_MODELS_DIR: <org>__<name>@<revision>."""
    return f"{doc['hf_repo'].replace('/', '__')}@{doc['revision']}"


def load_benchmark(key, root=ROOT):
    p = Path(root) / "benchmarks" / f"{key}.json"
    if not p.exists():
        raise RegistryError(f"unknown benchmark {key}")
    doc = read_json(p)
    _need(doc, BENCH_FIELDS, f"benchmark {key}")
    if f"{doc['id']}@{doc['version']}" != key:
        raise RegistryError(f"benchmark file {p.name} declares {doc['id']}@{doc['version']}")
    return Benchmark(doc["id"], doc["version"], doc)


def latest_rules(root=ROOT) -> "Rules":
    """The newest rules version (the one in force once its block is reached)."""
    return all_rules(root)[-1]


def load_rules_versions(base, root=ROOT):
    """The rules versions of one base's line, oldest first."""
    d = Path(root) / "rules" / base
    out = []
    for p in sorted(d.glob("*.json")):
        doc = read_json(p)
        _need(doc, RULES_FIELDS, f"rules {p}")
        if doc["base"] != base:
            raise RegistryError(f"{p} declares base {doc['base']}")
        out.append(Rules(doc))
    if not out:
        raise RegistryError(f"no rules for base {base}")
    return sorted(out, key=lambda r: r["effective_block"])


def all_rules(root=ROOT):
    """Every rules version of every base, in the order they take effect (no two share a block)."""
    out = [r for p in sorted((Path(root) / "rules").iterdir()) if p.is_dir() for r in load_rules_versions(p.name, root)]
    blocks = [r["effective_block"] for r in out]
    if len(set(blocks)) != len(blocks):
        raise RegistryError("two rules versions share an effective_block")
    return sorted(out, key=lambda r: r["effective_block"])


def rules_at(block, root=ROOT):
    """The rules version in force at `block` (latest effective_block <= block)."""
    best = None
    for r in all_rules(root):
        if r["effective_block"] <= block:
            best = r
    if best is None:
        raise RegistryError(f"no rules in force at block {block}")
    return best


def validate(root=ROOT):
    """Cross-check every rules version: bases and benchmarks exist, thresholds complete. Returns a list of errors."""
    try:
        versions = all_rules(root)
    except RegistryError as e:
        return [str(e)]
    errs = []
    for r in versions:
        at = f"{r['base']}@{r['effective_block']}"
        try:
            load_base(r["base"], root)
        except RegistryError as e:
            errs.append(f"{at}: {e}")
        for b in r["benchmarks"]:
            try:
                bench = load_benchmark(b["key"], root)
                params = {**bench.doc["judge"], **b.get("judge", {})}
                for k in ("eps_tie", "delta_win"):
                    if not isinstance(params.get(k), (int, float)):
                        errs.append(f"{at}: {b['key']} missing judge.{k}")
            except RegistryError as e:
                errs.append(f"{at}: {e}")
        if not (isinstance(r["fee"].get("usd_per_gpu_hour"), (int, float)) and r["fee"]["usd_per_gpu_hour"] > 0):
            errs.append(f"{at}: fee.usd_per_gpu_hour not set")
        run = r["run"]
        if not (isinstance(run.get("gpus"), int) and run["gpus"] > 0
                and str(run.get("gpu_model", "")).replace(" ", "").isalnum() and run.get("hours", 0) > 0):
            errs.append(f"{at}: run needs gpus, gpu_model, hours")
        if not (isinstance(run.get("max_agent_usd"), (int, float)) and run["max_agent_usd"] > 0):
            errs.append(f"{at}: run.max_agent_usd not set")
        for name, v in (("reveal_hours", r.doc.get("reveal_hours")), ("fee.review_usd", r["fee"].get("review_usd"))):
            if v is not None and not (isinstance(v, (int, float)) and v > 0):
                errs.append(f"{at}: {name} must be a positive number")
        if not str(r["fee"].get("treasury", "")).startswith("5"):
            errs.append(f"{at}: fee.treasury not set")
        if not r["review"].get("reviewers"):
            errs.append(f"{at}: review.reviewers is empty")
        j = r["judge"]
        if not 1 <= j["n_win"] <= len(r["benchmarks"]):
            errs.append(f"{at}: n_win {j['n_win']} out of range")
    return errs
