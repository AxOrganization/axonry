"""Execution-over-reflection steering at thought boundaries, written into the weights as one gated MLP unit.

  collect  --case C --sets S... --layer L --out data.npz   residual states at thought boundaries (execution vs
                                                            reflection thoughts) and MLP inputs (boundaries vs other
                                                            positions) of layer L, from the starting model's traces
  build    --case C --data data.npz --layer L --alpha A --name NAME   the edit spec (+ its firing check as JSON)

The unit must act only inside the thinking: tokens of the visible answer (after </think>) are negatives too.
A thought is a line of the thinking text (the model reasons in lines and bullets). It is a reflection thought when it
opens with a re-checking phrase (REFLECT below: "Wait", "Let me check", "Is there", "Alternatively", ...), otherwise an
execution thought. v = mean residual (output of layer L) at the boundary token before execution thoughts - the mean
before reflection thoughts (SEAL, arXiv 2504.07986, measured on the model's own traces).

The unit (wsp.gate's closed form, with a quantile threshold): it reads the layer's MLP input along the direction that
separates boundary tokens from all other tokens (LDA), fires above the `quantile` of the other tokens' scores, and
then adds about alpha * v to the residual stream. It replaces the layer's least-used MLP unit (smallest mean
|activation| on these traces). Values are written inline (set_rows / set_cols), so the spec is self-contained.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from wsp import case as wcase
from wsp import data, profiles
from wsp.common import read_jsonl, stable_seed, write_json

END = "</think>"
MAX_TOK = 12000
PER_TRACE = 40
REFLECT = re.compile(r"^(wait|but wait|hmm|let me (re-?)?(check|verify|double|re-?read|reconsider|think again)|"
                     r"double[- ]check|check|verify|re-?read|is there|could it|could there|alternatively|"
                     r"alternative|another (approach|way|interpretation|possibility)|however|actually|"
                     r"self-correction|re-?evaluat)", re.I)
BULLET = re.compile(r"^[\s*#>\-\d.)]+")


def thoughts(think):
    """[(char offset of the line, 'reflect' | 'execute')] for every line with at least 12 characters of text."""
    out, pos = [], 0
    for line in think.split("\n"):
        body = BULLET.sub("", line)
        if len(body) >= 12:
            out.append((pos, "reflect" if REFLECT.match(body) else "execute"))
        pos += len(line) + 1
    return out


def _sample(idx, k, key):
    idx = np.asarray(idx, dtype=int)
    if len(idx) <= k:
        return idx
    rng = np.random.default_rng(stable_seed(key) % (2 ** 32))
    return np.sort(rng.choice(idx, k, replace=False))


def cmd_collect(a):
    import torch  # noqa: PLC0415
    from wsp.probe import decoder, load_model  # noqa: PLC0415

    c = wcase.conf(a.case)
    prof = profiles.get(c["profile"])
    items = data.by_id()
    m, tok = load_model(wcase.model_path(a.case, wcase.BASE))
    _, layers = decoder(m)
    rows = [r for s in a.sets for r in read_jsonl(wcase.run_path(a.case, wcase.BASE, s))]
    rows.sort(key=lambda r: stable_seed(r["id"], r["k"], "seal"))
    caps = {}
    hooks = [layers[a.layer].register_forward_hook(
                 lambda mod, i, o: caps.__setitem__("res", o[0] if isinstance(o, tuple) else o)),
             layers[a.layer].mlp.register_forward_pre_hook(lambda mod, args: caps.__setitem__("x", args[0]))]
    E, R, B, O, used = [], [], [], [], 0
    absact = np.zeros(layers[a.layer].mlp.down_proj.in_features)
    hooks.append(layers[a.layer].mlp.down_proj.register_forward_pre_hook(
        lambda mod, args: caps.__setitem__("act", args[0])))
    ntok = 0
    try:
        for r in rows[: a.n]:
            think = r["text"].split(END)[0][:60000]
            th = thoughts(think)
            if len(th) < 10:
                continue
            answer = r["text"].split(END, 1)[1] if END in r["text"] and len(think) == len(r["text"].split(END)[0]) else ""
            p = tok.apply_chat_template([{"role": "user", "content": data.user_message(items[r["id"]])}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
            full = p + think + (END + answer if answer else "")
            enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"][-MAX_TOK:], enc["offset_mapping"][-MAX_TOK:]
            st = np.array([s for s, _ in offs])
            bpos = {}
            for c0, kind in th:          # the boundary token: the last token before the thought's first character
                i = int(np.searchsorted(st, len(p) + c0)) - 1
                if 0 <= i < len(ids):
                    bpos[i] = kind
            first = int(np.searchsorted(st, len(p)))
            quiet = r.get("source") in a.quiet_sources       # a task kind where the unit must stay silent
            ans0 = int(np.searchsorted(st, len(p) + len(think))) if answer else len(ids)   # the answer starts here
            ex = [] if quiet else _sample([i for i, k in bpos.items() if k == "execute"], PER_TRACE, f"{r['id']}e")
            rf = [] if quiet else _sample([i for i, k in bpos.items() if k == "reflect"], PER_TRACE, f"{r['id']}r")
            others = _sample([i for i in range(first, ans0) if i not in bpos], 2 * PER_TRACE, f"{r['id']}o")
            # the visible answer (after </think>): every token there, line breaks included, is a negative
            others = np.concatenate([others, _sample(list(range(ans0, len(ids))), PER_TRACE, f"{r['id']}a")]).astype(int)
            if quiet:   # its boundaries are negatives: the unit must not fire anywhere in such a task
                others = np.concatenate([others, _sample(list(bpos), 2 * PER_TRACE, f"{r['id']}q")]).astype(int)
            with torch.no_grad():
                m(input_ids=torch.tensor([ids], device=m.device), logits_to_keep=1, use_cache=False)
            res, x = caps["res"][0].float(), caps["x"][0].float()
            T = lambda ix: torch.as_tensor(np.asarray(ix, dtype=int), device=res.device)  # noqa: E731
            if len(ex):
                E.append(res[T(ex)].cpu().numpy())
            if len(rf):
                R.append(res[T(rf)].cpu().numpy())
            b = np.concatenate([np.asarray(ex, int), np.asarray(rf, int)]).astype(int)
            if len(b):
                B.append(x[T(b)].cpu().numpy())
            if len(others):
                O.append(x[T(others)].cpu().numpy())
            absact += caps["act"][0].float().abs().sum(0).cpu().numpy()
            ntok += len(ids)
            used += 1
    finally:
        for h in hooks:
            h.remove()
    arrays = {"E": np.concatenate(E), "R": np.concatenate(R), "B": np.concatenate(B), "O": np.concatenate(O),
              "use": absact / max(ntok, 1)}
    np.savez_compressed(a.out, **{k: v.astype(np.float32) for k, v in arrays.items()})
    v = arrays["E"].mean(0) - arrays["R"].mean(0)
    print(json.dumps({"traces": used, **{k: len(arrays[k]) for k in "ERBO"}, "v_norm": float(np.linalg.norm(v))}))


def cmd_build(a):
    from wsp.edits import decoder_prefix, spec_size, tensor_index  # noqa: PLC0415
    from wsp.gate import constant_direction, lda_direction, silu  # noqa: PLC0415

    z = np.load(a.data)
    v = z["E"].mean(0) - z["R"].mean(0)
    B, O = z["B"], z["O"]
    hb, ho = np.arange(len(B)) % 4 == 0, np.arange(len(O)) % 4 == 0      # every 4th position held out for the check
    Bf, Of = B[~hb], O[~ho]
    d = lda_direction(Bf, Of)
    so = Of @ d
    theta = float(np.quantile(so, a.quantile))
    mdir = constant_direction(np.concatenate([Bf, Of]))
    g = max(4.0, 12.0 / max(theta - float(np.median(so)), 1e-3))
    gate_row, up_row = g * (d - theta * mdir), 4.0 * mdir
    pre = Bf @ gate_row
    act = silu(pre) * (Bf @ up_row)
    amean = float(act[pre > 0].mean()) if (pre > 0).any() else 1.0
    down = a.alpha * v / amean
    c = wcase.conf(a.case)
    names = list(tensor_index(profiles.base_dir(c["profile"])))
    t = f"{decoder_prefix(names)}layers.{a.layer}.mlp."
    i = int(np.argmin(z["use"]))
    ops = [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [i], "values": [gate_row.astype(np.float32).tolist()]},
           {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [i], "values": [up_row.astype(np.float32).tolist()]},
           {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [i],
            "values": [[float(x)] for x in down.astype(np.float32)]}]
    edits = wcase.cdir(a.case) / "edits"
    edits.mkdir(parents=True, exist_ok=True)
    spec = {"name": a.name, "base": c["profile"], "ops": ops}
    if c.get("parent"):
        spec["parent"] = os.path.relpath(wcase.cdir(a.case) / c["parent"], edits)
    write_json(edits / f"{a.name}.json", spec)
    fb, fo = float((B[hb] @ gate_row > 0).mean()), float((O[ho] @ gate_row > 0).mean())
    print(json.dumps({"name": a.name, "layer": a.layer, "alpha": a.alpha, "unit": i, "fires_at_boundaries": fb,
                      "fires_elsewhere": fo, "ok": fb >= 0.9 and fo <= 0.02, "v_norm": float(np.linalg.norm(v)),
                      "payload": spec_size(edits / f"{a.name}.json")[1]}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--case", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n", type=int, default=240, help="traces read")
    p.add_argument("--quiet-sources", nargs="*", default=[],
                   help="practice sources whose traces only give negatives (the unit stays silent on such tasks)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--case", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    {"collect": cmd_collect, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
