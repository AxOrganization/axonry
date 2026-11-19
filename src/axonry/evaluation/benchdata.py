"""The rules' benchmark items, for every evaluation machine. The competition toolkit holds no benchmark (its
`bench:<name>` reads supplied files), so the orchestrator builds the items from their public sources
(benchmarks/sources.py, never on a run's machine), checks them against the frozen manifests with the toolkit, and
packs them as the toolkit reads them ($WSP_WORK/data/bench_<name>.jsonl). Only an evaluation machine receives the
pack; a run's machine never does.

  python3 -m axonry.evaluation.benchdata BENCHMARKS.json     on the evaluation machine: the items found match the
                                                             manifests (offline), exit 1 otherwise
  python3 -m axonry.evaluation.benchdata build               (stdin: the spec) write the items into $WSP_WORK/data and
                                                             print {key: items sha256}
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def spec(bench_docs) -> list:
    return [{"key": f"{b['id']}@{b['version']}", "task": b["task"], "manifest": b["items_manifest_sha256"]}
            for b in bench_docs]


def check(benchmarks) -> list:
    """-> the benchmarks whose items (loaded by the toolkit, in this process) differ from their manifests."""
    from wsp import data  # noqa: PLC0415
    bad = []
    for b in benchmarks:
        try:
            got = data.manifest(b["task"])["sha256"]
        except FileNotFoundError:
            bad.append(f"{b['key']}: items missing")
            continue
        if got != b["manifest"]:
            bad.append(f"{b['key']}: items {got[:12]} != frozen {b['manifest'][:12]}")
    return bad


def write_items(benchmarks) -> dict:
    """Build each benchmark's items into the toolkit's $WSP_WORK/data (in this process) -> {key: items sha256}."""
    from wsp import data  # noqa: PLC0415
    from wsp.common import write_jsonl  # noqa: PLC0415

    from ..benchmarks import sources  # noqa: PLC0415
    out = {}
    for b in benchmarks:
        name = b["task"].split(":", 1)[1]
        write_jsonl(data.DATA / f"bench_{name}.jsonl", sources.rows(name))
        out[b["key"]] = data.manifest(b["task"])["sha256"]
    return out


def _built(competition: Path, s: list, work: str) -> dict:
    """write_items in a child process importing the pinned toolkit -> {key: items sha256}."""
    r = subprocess.run([sys.executable, "-m", "axonry.evaluation.benchdata", "build"], input=json.dumps(s),
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(Path(competition).resolve()), "WSP_WORK": work})
    if r.returncode != 0:
        raise RuntimeError(f"benchmark items: {(r.stdout + r.stderr).strip()[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def manifests(competition: Path, tasks: list) -> dict:
    """{task: items sha256} as built now (registry pin-bench)."""
    with tempfile.TemporaryDirectory() as work:
        got = _built(competition, [{"key": t, "task": t} for t in tasks], work)
    return got


def _holds(pack: Path, s: list) -> bool:
    """A pack built before holds exactly the files the toolkit reads now (one per benchmark): one in an older layout
    is built again."""
    want = {"data"} | {f"data/bench_{b['task'].split(':', 1)[1]}.jsonl" for b in s}
    try:
        with tarfile.open(pack) as tf:
            return set(tf.getnames()) == want
    except (OSError, tarfile.TarError):
        return False


def build(competition: Path, bench_docs, dest_dir: Path) -> Path:
    """-> benchdata-<sha>.tar (data/bench_*.jsonl), built once per set of benchmarks, only when every benchmark's
    items match its frozen manifest."""
    s = spec(bench_docs)
    out = Path(dest_dir) / f"benchdata-{hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]}.tar"
    if out.exists() and _holds(out, s):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        got = _built(competition, s, work)
        bad = [f"{b['key']}: items {got[b['key']][:12]} != frozen {b['manifest'][:12]}" for b in s
               if got[b["key"]] != b["manifest"]]
        if bad:
            raise RuntimeError("benchmark items: " + "; ".join(bad))
        tmp = out.with_suffix(".tmp")
        with tarfile.open(tmp, "w") as tf:
            tf.add(Path(work) / "data", arcname="data")
        tmp.replace(out)
    return out


def main(argv=None):
    arg = (argv or sys.argv[1:])[0]
    if arg == "build":
        print(json.dumps(write_items(json.loads(sys.stdin.read()))))
        return 0
    bad = check(json.loads(Path(arg).read_text()))
    if bad:
        print("; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
