"""The storage layout: the only place object keys are spelled.

Public bucket (served read-only through the public domain) and private bucket. Every other module asks this module
for a key, so a path never drifts between writer, reader, validator and miner CLI.

Public bucket: mutable files are the live views (weights.json, index.json, status.json, models/<base>/line.json);
everything else is written once and kept for good, so any decision can be recomputed later.
"""

# ---- public bucket -------------------------------------------------------------------------------------------
WEIGHTS = "weights.json"          # fixed forever: the validator's only input
INDEX = "index.json"              # fixed forever: discovery entry point
STATUS = "status.json"


def base_def(base_id: str) -> str:
    return f"bases/{base_id}.json"


def benchmark_def(bench_id: str, version: int) -> str:
    return f"benchmarks/{bench_id}@{version}.json"


def benchmark_items(bench_id: str, version: int) -> str:
    """A benchmark's items, public (write-once): exactly what its frozen manifest pins."""
    return f"benchmarks/{bench_id}@{version}/items.jsonl"


def rules_version(rules_sha256: str) -> str:
    return f"rules/{rules_sha256}.json"


def line(base_id: str) -> str:
    """A base model's line: its entries, champions and the base model's own scores (one summary per base)."""
    return f"models/{base_id}/line.json"


def model_dir(model_id: str) -> str:
    """Where a model's own files live: an entry (its sid) or a base model ("base:<id>", the reference until its line
    has a champion)."""
    if model_id.startswith("base:"):
        return f"models/{model_id[5:]}/base/"
    return f"submissions/{model_id}/"


def submission(sid: str, name: str = "") -> str:
    return model_dir(sid) + name


def results(model_id: str, bench_key: str) -> str:
    return f"{model_dir(model_id)}results/{bench_key}.jsonl.gz"


def score(model_id: str) -> str:
    return model_dir(model_id) + "score.json"


def submission_edit_prefix(sid: str) -> str:
    return f"submissions/{sid}/edit/"


def submission_agent_prefix(sid: str) -> str:
    return f"submissions/{sid}/agent/"


# ---- private bucket ------------------------------------------------------------------------------------------
def staging_prefix(sid: str) -> str:
    """The only prefix a run's temporary R2 credentials may touch."""
    return f"staging/{sid}/"


STATE_PREFIX = "state/"

IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
MUTABLE_CACHE = "public, max-age=60"
MUTABLE_PUBLIC = {WEIGHTS, INDEX, STATUS}


def is_mutable(key: str) -> bool:
    if key in MUTABLE_PUBLIC:
        return True
    parts = key.split("/")
    return len(parts) == 3 and parts[0] == "models" and parts[2] == "line.json"
