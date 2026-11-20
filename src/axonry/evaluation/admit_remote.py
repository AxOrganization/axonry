#!/usr/bin/env python3
"""Admission check of an edit spec against the competition protocol (WeightSpaceProgramming rules.json,
section "edit"). The competition repo defines the limits; checking them is this project's job. Runs where
WeightSpaceProgramming (wsp) is importable and the verified base is present: on the evaluation machine (admission) and
on the run's machine (`axonry-check`, the agent's self-check) — one code path for both.

  python admit_remote.py --spec-dir DIR --base-dir BASE --base-id ID [--build-dir OUT]

Prints {"passed", "violations": [...], "info": {...}, "model_digest", "model_dir"}; exit code 0 iff passed.
Checks: file types; edit.json present and its base == the submission's base; REPORT.md present; ops whitelisted;
tensors editable (2-D .weight matrices of the main decoder stack); payload bytes within edit.limits (this spec's own
ops, wsp.edits.spec_size; the number of tensors is not limited); then the spec (with its parent chain) is built on the verified base and the
relative change of its own layer vs the parent model is checked per tensor and globally. The no-overfitting rule
is normative: admission enforces the limits, it does not judge how the spec was obtained.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

FORBIDDEN = re.compile(r"norm|embed|lm_head|bias")


def load_rules():
    from wsp.rules import RULES  # noqa: PLC0415
    return RULES


def edit_limits(rules):
    """The one set of limits that binds every edit (competition rules.json -> edit.limits)."""
    return rules["edit"]["limits"]


def tensor_violation(name, prefix, ndim=None):
    if not re.match(re.escape(prefix) + r"layers\.\d+\.", name):
        return "not inside a main decoder layer"
    if not name.endswith(".weight") or FORBIDDEN.search(name):
        return "not an editable weight matrix (embedding / head / norm / bias)"
    if ndim is not None and ndim != 2:
        return f"{ndim}-D parameter"
    return None


def editable_norm(base_dir, prefix):
    """Frobenius norm over all editable matrices of the base (denominator of the global relative change)."""
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import tensor_index  # noqa: PLC0415

    idx, sq = tensor_index(base_dir), 0.0
    for shard in sorted(set(idx.values())):
        with safe_open(str(Path(base_dir) / shard), "pt") as f:
            for n in f.keys():
                if tensor_violation(n, prefix) is None:
                    t = f.get_tensor(n)
                    if t.dim() == 2:
                        sq += float(t.float().pow(2).sum())
    return sq ** 0.5


def model_digest(man):
    h = hashlib.sha256()
    for t in sorted(man["tensors"], key=lambda t: t["name"]):
        h.update(f"{t['name']}\0{t['edited_sha256']}\n".encode())
    return h.hexdigest()


def static_checks(spec_dir, spec, rules, prefix=None, ndims=None):
    """Checks that need no weights (prefix/ndims: from the base's tensor index when available)."""
    out = []
    e = rules["edit"]
    allowed = {f.removeprefix("edit/") for f in rules["upload"]["files"]}   # what an upload holds (and each parent)
    for p in Path(spec_dir).rglob("*"):
        placed = p.parent != Path(spec_dir)     # in a published start placed beside the spec: an archived one keeps
        if (p.is_file() and "__pycache__" not in p.parts and p.name not in allowed  # its lineage note (FLAT.json)
                and not (placed and p.name == "FLAT.json")):
            out.append(f"{p.relative_to(spec_dir)}: not an upload file ({', '.join(sorted(allowed))})")
    required = [f.removeprefix("edit/") for f in rules["upload"]["required"]]
    out += [f"missing {f}" for f in required if not (Path(spec_dir) / f).exists()]
    lim = edit_limits(rules)
    ops = spec.get("ops")
    if not isinstance(ops, list) or not ops:
        return out + ["edit.json: no ops"], {}
    for op in ops:
        if op.get("op") not in e["ops"]:
            out.append(f"op {op.get('op')} not allowed")
        if prefix is not None:
            if op.get("tensor") not in (ndims or {}):
                out.append(f"{op.get('tensor')}: no such tensor in the base")
            elif why := tensor_violation(op["tensor"], prefix, ndims[op["tensor"]]):
                out.append(f"{op['tensor']}: {why}")
    try:
        from wsp.edits import spec_size  # noqa: PLC0415  (the toolkit's own count: this spec's ops, not its parents)
        n, pb = spec_size(Path(spec_dir) / "edit.json")
    except Exception as ex:
        return out + [f"edit arrays unreadable: {type(ex).__name__}: {ex}"], {}
    if pb > lim["max_payload_bytes"]:
        out.append(f"payload {pb} bytes > {lim['max_payload_bytes']}")
    return out, {"tensors": n, "payload_bytes": pb}


def magnitude_checks(man, lim, total_norm):
    """Limits apply to this spec's own layer only (stacked parents are not counted, the chain is not limited):
    per tensor relative_change_vs_parent; global sqrt(sum (rel_vs_parent * parent_norm)^2) / editable base norm."""
    out = []
    own = set(man.get("own_tensors") or [t["name"] for t in man["tensors"]])
    mine = [t for t in man["tensors"] if t["name"] in own]
    for t in mine:
        rc = t.get("relative_change_vs_parent", t["relative_change"])
        if rc > lim["max_relative_change_per_tensor"]:
            out.append(f"{t['name']}: relative change {rc:.3f} > {lim['max_relative_change_per_tensor']}")
    g = (sum((t.get("relative_change_vs_parent", t["relative_change"]) * t.get("parent_norm", t["base_norm"])) ** 2
             for t in mine) ** 0.5) / total_norm
    if g > lim["max_global_relative_change"]:
        out.append(f"global relative change {g:.4f} > {lim['max_global_relative_change']}")
    return out, g


def admit(spec_dir, base_dir, base_id, build_dir=None, rules=None):
    from safetensors import safe_open  # noqa: PLC0415
    from wsp.edits import build, decoder_prefix, tensor_index  # noqa: PLC0415

    rules = rules or load_rules()
    spec_dir = Path(spec_dir)
    res = {"passed": False, "violations": [], "info": {}, "model_digest": None, "model_dir": None}
    if not (spec_dir / "edit.json").exists():
        res["violations"].append("missing edit.json")
        return res
    try:
        spec = json.loads((spec_dir / "edit.json").read_text())
    except ValueError as e:
        res["violations"].append(f"edit.json: {e}")
        return res
    if spec.get("base") != base_id:
        res["violations"].append(f"edit.json base {spec.get('base')!r} != submission base {base_id!r}")
        return res
    idx = tensor_index(base_dir)
    prefix = decoder_prefix(idx)
    ndims = {}
    for op in spec.get("ops") or []:
        n = op.get("tensor")
        if n in idx and n not in ndims:
            with safe_open(str(Path(base_dir) / idx[n]), "pt") as f:
                ndims[n] = len(f.get_slice(n).get_shape())
    v, info = static_checks(spec_dir, spec, rules, prefix, {**{k: None for k in idx}, **ndims})
    res["violations"] += v
    res["info"].update(info)
    if res["violations"]:
        return res
    work = Path(tempfile.mkdtemp(prefix="axonry-admit-"))
    try:
        out_dir = Path(build_dir) if build_dir else work / "model"
        try:
            _, man = build(spec_dir / "edit.json", out_dir, base_dir=base_dir)
        except Exception as e:
            res["violations"].append(f"build: {type(e).__name__}: {e}")
            return res
        v, g = magnitude_checks(man, edit_limits(rules), editable_norm(base_dir, prefix))
        res["violations"] += v
        res["info"]["global_relative_change"] = g
        res["info"]["chain"] = man.get("chain")
        res["info"]["tensors_detail"] = [{k: t.get(k) for k in ("name", "relative_change", "relative_change_vs_parent",
                                                                 "changed_entries")} for t in man["tensors"]]
        res["model_digest"] = model_digest(man)
        res["model_dir"] = str(out_dir) if build_dir else None
        res["passed"] = not res["violations"]
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="admission check of an edit spec")
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--base-id", required=True)
    ap.add_argument("--build-dir")
    a = ap.parse_args(argv)
    res = admit(a.spec_dir, a.base_dir, a.base_id, a.build_dir)
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
