"""Runs on an evaluation machine: a champion's archive form, its whole chain as one edit on the base.

python -m axonry.evaluation.flatten_remote --edit edit --model model --base-dir base --out flat

wsp.edits.flatten_chain carries every link's ops into one spec, rounding between links as building the chain does, so
it builds the same model bit for bit. That is checked here against the model evaluated (built from the chain by
admission on this machine): the flat edit is built and every tensor compared; only an identical build is written out.
Prints {"ok", "ops", "bytes", "problem"}.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _tensors(model_dir):
    from safetensors.torch import load_file  # noqa: PLC0415
    from wsp.edits import tensor_index  # noqa: PLC0415
    out = {}
    for shard in sorted(set(tensor_index(model_dir).values())):
        out.update(load_file(str(Path(model_dir) / shard)))
    return out


def flatten(edit_dir, model_dir, base_dir, out_dir) -> dict:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from wsp.common import write_json  # noqa: PLC0415
    from wsp.edits import build, flatten_chain  # noqa: PLC0415

    spec, arrays = flatten_chain(Path(edit_dir) / "edit.json")
    out = Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if arrays:
        np.savez(out / "edit.npz", **arrays)
    write_json(out / "edit.json", spec)
    with tempfile.TemporaryDirectory() as tmp:
        build(out / "edit.json", Path(tmp) / "model", base_dir=base_dir)
        rebuilt, model = _tensors(Path(tmp) / "model"), _tensors(model_dir)
        bad = sorted(n for n in model.keys() | rebuilt.keys()
                     if n not in model or n not in rebuilt or not torch.equal(rebuilt[n], model[n]))
    if bad:
        shutil.rmtree(out, ignore_errors=True)
        return {"ok": False, "problem": f"the flat edit builds another model: {len(bad)} tensors differ, e.g. {bad[0]}"}
    return {"ok": True, "ops": len(spec["ops"]), "bytes": sum(p.stat().st_size for p in out.iterdir())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="a champion's chain as one edit on the base, checked identical")
    for k in ("--edit", "--model", "--base-dir", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    res = flatten(a.edit, a.model, a.base_dir, a.out)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
