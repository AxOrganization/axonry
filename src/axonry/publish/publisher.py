"""The single writer of the public bucket. Dependencies are written before what references them:
  edit/ -> admission.json -> results/ -> score.json -> submission.json -> models/<base>/line.json
  -> weights.json -> index.json
Immutable objects are write-once (store enforces); weights.json carries a monotonic sequence and the owner signature.
"""
import gzip
import json
import time
from pathlib import Path

from .. import registry
from ..common import layout
from ..common.canon import pretty
from ..common.sign import sign_doc, verify_doc

SCHEMAS = {"weights": "axonry.weights/1", "score": "axonry.score/1", "submission": "axonry.submission/1",
           "admission": "axonry.admission/1", "line": "axonry.line/1", "index": "axonry.index/1",
           "status": "axonry.status/1", "ruling": "axonry.ruling/1"}


class PublishError(RuntimeError):
    pass


def _json(store, key, doc):
    store.put(key, pretty(doc))


def publish_registry(store, root=registry.ROOT):
    """bases/, benchmarks/ and every rules version (content-addressed; index.json lists them)."""
    root = Path(root)
    for p in sorted((root / "bases").glob("*.json")):
        store.put(layout.base_def(p.stem), p.read_bytes())
    for p in sorted((root / "benchmarks").glob("*.json")):
        bid, _, v = p.stem.partition("@")
        store.put(layout.benchmark_def(bid, int(v)), p.read_bytes())
    for r in registry.all_rules(root):
        _json(store, layout.rules_version(r.sha256), r.doc)


def publish_dir(store, prefix: str, local: Path, content_type=None):
    """Upload every file under local to prefix (follows symlinks: built models link untouched base shards)."""
    n = 0
    for p in sorted(Path(local).rglob("*")):
        if p.is_file():
            ct = content_type or ("application/json" if p.suffix == ".json" else "application/octet-stream")
            store.put_file(prefix + p.relative_to(local).as_posix(), p, ct)
            n += 1
    return n


def publish_results(store, model_id, bench_key, rows_jsonl: bytes):
    """Per item and sample, for an entry or a base model alike: what judging and any re-computation need (id, sample,
    correct, finish, tokens), without the generated text, which would make results orders of magnitude larger than
    the decision they support."""
    rows = [{k: v for k, v in json.loads(line).items() if k != "text"} for line in rows_jsonl.split(b"\n") if line.strip()]
    data = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()
    store.put(layout.results(model_id, bench_key), gzip.compress(data, mtime=0), "application/gzip")


def publish_admission(store, sid, admission: dict):
    _json(store, layout.submission(sid, "admission.json"), {"schema": SCHEMAS["admission"], "sid": sid, **admission})


def publish_score(store, model_id, score: dict):
    """score.json of an entry (its match: per benchmark its accuracy and the reference's, thresholds, verdict) or of a
    base model (its accuracy per benchmark: the reference of its line until a first champion)."""
    _json(store, layout.score(model_id), {"schema": SCHEMAS["score"], "model": model_id, **score})


def publish_submission(store, sid, record: dict):
    _json(store, layout.submission(sid, "submission.json"), {"schema": SCHEMAS["submission"], "sid": sid, **record})


def publish_ruling(store, sid, ruling: dict):
    """submissions/<sid>/ruling.json: the owner's disqualification of an evaluated submission (write-once)."""
    _json(store, layout.submission(sid, "ruling.json"), {"schema": SCHEMAS["ruling"], "sid": sid, **ruling})


def publish_line(store, base, line: dict):
    """models/<base>/line.json: everything a leaderboard of this base model needs in one file - the champion, its
    succession, the base model's own accuracy, the Pareto front and one row per entry (status, decision, reference
    and per-benchmark scores); each row's full detail lives under submissions/<sid>/."""
    _json(store, layout.line(base), {"schema": SCHEMAS["line"], "base": base, "updated_at": int(time.time()), **line})


def build_weights(netuid, lines, sequence, block, fallback_uid=0, burn=0.0):
    """lines: the lines in force, the primary one first, each {"base", "share", "champion": {"sid", "hotkey", "uid"} or
    None}. A line's share goes to its champion, or to the fallback uid0 (burn) while it has none (or it is not
    registered); burn: the owner's share of the weight burned (to uid0) beside the champions'. The top-level base and
    champion are the primary line's."""
    weights, burned = [], 0.0
    for x in lines:
        c = x["champion"]
        if c and c.get("uid") is not None and burn < 1:
            weights.append({"uid": c["uid"], "hotkey": c["hotkey"], "weight": round(x["share"] * (1.0 - burn), 6)})
            burned += x["share"] * burn
        else:
            burned += x["share"]
    if burned > 1e-9 or not weights:
        weights.append({"uid": fallback_uid, "hotkey": None, "weight": round(burned, 6) if weights else 1.0})
    return {"schema": SCHEMAS["weights"], "netuid": netuid, "sequence": sequence, "base": lines[0]["base"],
            "issued_at": int(time.time()), "issued_block": block, "champion": lines[0]["champion"], "burn": burn,
            "lines": lines, "weights": weights,
            "fallback": {"uid": fallback_uid, "reason": "no champion or champion hotkey not registered: uid0 (burn)"}}


def weight_lines(doc) -> list:
    """The lines a weights.json was issued for (one written before lines existed: its one base, share 1)."""
    return doc.get("lines") or [{"base": doc.get("base"), "share": 1.0, "champion": doc.get("champion")}]


def publish_weights(store, doc: dict, keypair):
    old = store.get(layout.WEIGHTS)
    if old:
        prev = json.loads(old)
        if prev.get("sequence", -1) >= doc["sequence"]:
            raise PublishError(f"weights sequence {doc['sequence']} not above published {prev.get('sequence')}")
    signed = sign_doc(doc, keypair)
    _json(store, layout.WEIGHTS, signed)
    back = json.loads(store.get(layout.WEIGHTS))
    if back != signed or not verify_doc(back, keypair.ss58_address):
        raise PublishError("weights.json read-back does not verify")
    return signed


def index_lines(lines) -> list:
    return [{"base": r["base"], "share": registry.share(r), "rules": layout.rules_version(r.sha256)} for r in lines]


def publish_index(store, lines, root=registry.ROOT):
    """index.json: the entry point - the lines in force (base, share of the weight, rules version; the primary one
    also at top level), and every rules version (base, effective block, key), so each base's line, past ones included,
    can be found without listing the bucket. Netuid, owner and this location itself are published with the code
    (axonry_validator.network), not repeated here."""
    _json(store, layout.INDEX, {"schema": SCHEMAS["index"], "base": lines[0]["base"],
                                "rules": layout.rules_version(lines[0].sha256), "lines": index_lines(lines),
                                "versions": [{"base": r["base"], "effective_block": r.doc.get("effective_block"),
                                              "rules": layout.rules_version(r.sha256)}
                                             for r in registry.all_rules(root)]})