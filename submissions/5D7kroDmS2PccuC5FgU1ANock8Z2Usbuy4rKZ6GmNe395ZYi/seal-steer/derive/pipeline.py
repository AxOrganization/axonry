"""The whole method as one resumable pipeline: one job per GPU, every GPU busy, deadline-aware, uploads by itself.

Run it detached in the sandbox and check on it; rerunning resumes (finished steps are skipped):
  tmux new -d -s seal "python <submission>/derive/pipeline.py run --parent <parent/edit/edit.json> --task <task.json> \
      2>&1 | tee -a /workspace/work/pipeline.log"
  python <submission>/derive/pipeline.py status

Stages (reasons in METHOD.md):
  S1 generate  the starting model on practice problems of four sources: dev128 (2 samples: the pairing reference),
               the rest of DEV as two harvest sets (1 sample), holdout256 (1 sample; used only after freezing)
  S2 collect   thought-boundary states at layers LAYERS (one GPU each), from the harvest and dev traces; the
               boundaries of instruction-following traces (QUIET) are negatives, so the unit stays silent there
  S3 build     one gated steering unit per (layer, alpha); a candidate is kept only if its unit fires on >= 90% of
               held-out boundaries and <= 2% of other held-out tokens
  S4 generate  the kept candidates on dev128 (2 samples), paired with the starting model
  S5 decide    best overall paired gain among those where no practice source loses more than SOURCE_FLOOR points;
               the two best are re-checked on ifcheck (128 more instruction-following problems, 2 samples): a
               candidate losing more than IF_FLOOR points there is out; upload the best one left at once
  S6 refine    the best layer at alpha x0.7 and x1.4, and the best two layers' units together -> upload if better
  S7 confirm   the chosen edit on holdout256; final report; final upload
Fewer than MIN_LEFT minutes left at any point: straight to the final upload of the best result so far.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AXONRY_WORK", "/workspace/work"))
os.environ.setdefault("WSP_CASES", str(WORK / "cases"))
os.environ.setdefault("WSP_WORK", str(WORK / "wsp"))
STATE = WORK / "pipeline"
CASE = "seal"
PROFILE = "qwen35-0.8b-think"
SOURCES = ("gsm8k_train", "math_train", "mmlupro", "ifsynth")
LAYERS = (16, 20)
QUIET = ("ifsynth",)    # instruction-following practice: the unit must stay silent on such tasks
IF_FLOOR = -2.0         # points: the largest loss on the instruction-following check a candidate may have
ALPHAS = (0.5, 1.0, 2.0)
SOURCE_FLOOR = -3.0
MIN_LEFT = 45 * 60
PY = sys.executable

from wsp import case as wcase  # noqa: E402  (after the WSP_* paths are set)
from wsp.common import read_json, read_jsonl, write_json  # noqa: E402


# ------------------------------------------------------------------------------------------- plumbing
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATE / "events.log", "a") as f:
        f.write(line + "\n")


def time_left():
    p = Path("/etc/axonry-deadline")
    return int(p.read_text().strip()) - time.time() if p.exists() else 10 ** 9


def late():
    return time_left() < MIN_LEFT


def progress(status, details=""):
    """PROGRESS.md: its first line is the public one-sentence status (at most 20 words)."""
    (WORK / "PROGRESS.md").write_text(" ".join(status.split()[:20]) + "\n\n" + details + "\n\nEvents: "
                                      + str(STATE / "events.log") + "\n")


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return list(range(sum(1 for l in out.splitlines() if l.startswith("GPU")))) or [0]
    except Exception:  # noqa: BLE001
        return [0]


def jobs(batch, critical=True):
    """batch: [(name, argv)], one per GPU at a time, each in its own process group; finished ones are skipped."""
    done, logs = STATE / "done", STATE / "logs"
    done.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    todo = [(n, a) for n, a in batch if not (done / n).exists()]
    free, running, failed = gpus(), {}, []
    while todo or running:
        while todo and free:
            n, argv = todo.pop(0)
            g = free.pop(0)
            running[n] = (subprocess.Popen(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                           stdout=open(logs / f"{n}.log", "a"), stderr=subprocess.STDOUT,
                                           start_new_session=True), g)
            log(f"start {n} on GPU {g}")
        time.sleep(15)
        for n, (p, g) in list(running.items()):
            if p.poll() is None:
                continue
            del running[n]
            free.append(g)
            if p.returncode == 0:
                (done / n).write_text(str(int(time.time())))
                log(f"done  {n}")
            else:
                failed.append(n)
                log(f"FAILED {n} (exit {p.returncode}); log {logs / f'{n}.log'}")
    if failed and critical:
        progress(f"Stopped: {failed[0]} failed; fixing and resuming", f"Failed: {failed}")
        raise SystemExit(f"failed: {failed}")
    return failed


def one(name, argv):
    mark = STATE / "done" / name
    if mark.exists():
        return (STATE / "logs" / f"{name}.log").read_text()
    mark.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(argv, capture_output=True, text=True)
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    (STATE / "logs" / f"{name}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"{name} failed: {(r.stderr or r.stdout)[-800:]}")
    mark.write_text(str(int(time.time())))
    return r.stdout


def wsp_case(*args):
    return [PY, "-m", "wsp.case", *args]


def seal(*args):
    return [PY, str(HERE / "seal.py"), *args]


def notes(text):
    p = wcase.cdir(CASE) / "NOTES.md"
    old = p.read_text() if p.exists() else f"# Decisions: {CASE}\n\n"
    p.write_text(old + text.rstrip() + "\n\n")


# ------------------------------------------------------------------------------------------- measurement
def _paired(base, rows):
    b = {(r["id"], r["k"]): r for r in base}
    per = {}
    for r in rows:
        if (r["id"], r["k"]) in b:
            per.setdefault(r["id"], []).append(int(r["correct"]) - int(b[(r["id"], r["k"])]["correct"]))
    pm = [sum(v) / len(v) for v in per.values()]
    if not pm:
        return None
    g = sum(pm) / len(pm)
    se = math.sqrt(sum((x - g) ** 2 for x in pm) / max(len(pm) - 1, 1) / len(pm))
    cb = sum(b[(r["id"], r["k"])]["cut"] for r in rows if (r["id"], r["k"]) in b) / len(rows)
    return {"gain": round(100 * g, 2), "se": round(100 * se, 2), "problems": len(pm),
            "cut": [round(100 * cb, 1), round(100 * sum(r["cut"] for r in rows) / len(rows), 1)]}


def compare(models, set_name):
    """{model: {"all": paired, <source>: paired}} against the starting model on the same problems and seeds."""
    base = read_jsonl(wcase.run_path(CASE, wcase.BASE, set_name))
    out = {}
    for m in models:
        p = wcase.run_path(CASE, m, set_name)
        if p.exists():
            rows = read_jsonl(p)
            out[m] = {"all": _paired(base, rows), **{
                s: _paired([r for r in base if r.get("source") == s], [r for r in rows if r.get("source") == s])
                for s in sorted({r.get("source") for r in base})}}
    return out


def pick_best(models):
    res = compare(models, "dev128")
    write_json(wcase.cdir(CASE) / "results" / "compare__dev128.json", res)
    ok = {m: r for m, r in res.items() if r["all"] and r["all"]["gain"] > 0
          and all(v is None or v["gain"] >= SOURCE_FLOOR for k, v in r.items() if k != "all")}
    best = max(ok, key=lambda m: ok[m]["all"]["gain"]) if ok else None
    return best, res


# ------------------------------------------------------------------------------------------- upload
def upload(edit, task, why):
    if not task:
        log(f"(rehearsal, no --task) would upload {edit}: {why}")
        return False
    subprocess.run(wsp_case("report"), capture_output=True)
    pk = subprocess.run([PY, "-m", "wsp.upload", CASE, edit, "--dry-run"], capture_output=True, text=True)
    if pk.returncode:
        log(f"package {edit} failed: {pk.stderr[-500:]}")
        return False
    out = Path(os.environ["WSP_WORK"]) / "deliver" / f"{CASE}__{edit}" / "edit"
    chk = subprocess.run(["axonry-check", str(out), "--task", task], capture_output=True, text=True)
    if chk.returncode:
        log(f"axonry-check rejects {edit}: {chk.stdout[-800:]}{chk.stderr[-300:]}")
        return False
    up = subprocess.run([PY, "-m", "wsp.upload", CASE, edit, "--task", task], capture_output=True, text=True)
    if up.returncode:
        log(f"upload {edit} failed: {up.stderr[-500:]}")
        return False
    log(f"UPLOADED {edit} ({why})")
    return True


def best_so_far():
    p = STATE / "best.json"
    return read_json(p) if p.exists() else None


def keep_best(edit, gain, why, task):
    write_json(STATE / "best.json", {"edit": edit, "dev128_gain": gain})
    notes(f"Best so far: {edit} ({gain:+.1f} points on dev128, {why}).")
    upload(edit, task, why)


# ------------------------------------------------------------------------------------------- stages
def s0_init(parent):
    if (wcase.cdir(CASE) / "case.json").exists():
        return
    subprocess.run(wsp_case("init", CASE, "--profile", PROFILE, *(["--parent", parent] if parent else []),
                            "--practice", *SOURCES, "--dev", "256", "--holdout", "512",
                            "--subset", "dev128=dev:128", "--subset", "holdout256=holdout:256"), check=True)
    dev, d128 = wcase.set_ids(CASE, "dev"), set(wcase.set_ids(CASE, "dev128"))
    rest = [i for i in dev if i not in d128]
    write_json(wcase.cdir(CASE) / "sets" / "harvest_a.json", [i for i in rest if i.startswith(("gsm8k", "math"))])
    write_json(wcase.cdir(CASE) / "sets" / "harvest_b.json", [i for i in rest if i.startswith(("mmlupro", "ifsynth"))])
    from wsp import data  # noqa: PLC0415
    used = set(dev) | set(wcase.set_ids(CASE, "holdout"))
    write_json(wcase.cdir(CASE) / "sets" / "ifcheck.json",
               [r["id"] for r in data.load("practice:ifsynth") if r["id"] not in used][:128])
    notes(("Starting model: the parent edit (the champion). " if parent else "Starting model: the base. ")
          + "Practice sources: " + ", ".join(SOURCES) + ". DEV 256 (dev128 for selection), HOLDOUT 512 (holdout256 "
          "only after freezing).")


def build_candidates(sd):
    """S3: every (layer, alpha); keep the ones whose unit fires where it should."""
    kept = []
    for layer in LAYERS:
        for alpha in ALPHAS:
            name = f"SEAL{layer}_a{alpha:g}"
            info = json.loads(one(f"build_{name}", seal("build", "--case", CASE, "--data", str(sd / f"seal_{layer}.npz"),
                                                        "--layer", str(layer), "--alpha", str(alpha),
                                                        "--name", name)).strip().splitlines()[-1])
            if info["ok"]:
                kept.append(name)
            notes(f"{name}: unit {info['unit']} fires on {info['fires_at_boundaries']:.0%} of held-out boundaries, "
                  f"{info['fires_elsewhere']:.1%} of other tokens; |v| {info['v_norm']:.2f}; "
                  f"{'kept' if info['ok'] else 'dropped'}.")
    return kept


def eligible(res):
    """Candidates with a positive overall gain and no practice source below SOURCE_FLOOR, best first."""
    ok = [m for m, r in res.items() if r["all"] and r["all"]["gain"] > 0
          and all(v is None or v["gain"] >= SOURCE_FLOOR for k, v in r.items() if k != "all")]
    return sorted(ok, key=lambda m: -res[m]["all"]["gain"])


def if_guard(res, task, top=4):
    """The best `top` eligible candidates on ifcheck (one per GPU, at once); -> the best one losing at most IF_FLOOR
    points there. Several are checked because a candidate silent on such tasks can still lose 2 points by chance."""
    cands = eligible(res)[:top]
    if not cands:
        return None
    jobs([(f"gen_{m}_ifcheck", wsp_case("run", CASE, m, "ifcheck", "--n", "2")) for m in cands], critical=False)
    chk = compare(cands, "ifcheck")
    notes("Instruction-following check (ifcheck, 2 samples):\n```\n" + json.dumps(chk, indent=1) + "\n```")
    for m in cands:
        if chk.get(m, {}).get("all") and chk[m]["all"]["gain"] >= IF_FLOOR:
            return m
        log(f"{m} fails the instruction-following check: {chk.get(m, {}).get('all')}")
    return None


def merge(a_name, b_name, name):
    """Both units in one spec (they sit in different layers)."""
    ed = wcase.cdir(CASE) / "edits"
    a, b = read_json(ed / f"{a_name}.json"), read_json(ed / f"{b_name}.json")
    spec = {**a, "name": name, "ops": a["ops"] + b["ops"]}
    write_json(ed / f"{name}.json", spec)
    return name


def run(a):
    STATE.mkdir(parents=True, exist_ok=True)
    s0_init(a.parent)
    sd = wcase.scratch(CASE)
    sd.mkdir(parents=True, exist_ok=True)

    if not late():
        progress("Generating the starting model's answers on practice problems (dev, harvest, holdout) on all GPUs")
        jobs([("gen_base_dev128", wsp_case("run", CASE, "base", "dev128", "--n", "2")),
              ("gen_base_harvest_a", wsp_case("run", CASE, "base", "harvest_a")),
              ("gen_base_harvest_b", wsp_case("run", CASE, "base", "harvest_b")),
              ("gen_base_holdout256", wsp_case("run", CASE, "base", "holdout256")),
              ("gen_base_ifcheck", wsp_case("run", CASE, "base", "ifcheck", "--n", "2"))])
        base = read_json(wcase.cdir(CASE) / "results" / "base__dev128.json")["summary"]
        notes(f"Starting model on dev128 (2 samples): accuracy {base['acc']:.3f}, cut at the token limit "
              f"{base['cut_share']:.3f}, kinds {base['kinds']}.")

    if not late():
        progress("Measuring execution vs reflection states at thought boundaries in three layers")
        jobs([(f"collect_{l}", seal("collect", "--case", CASE, "--sets", "harvest_a", "harvest_b", "dev128",
                                    "--layer", str(l), "--quiet-sources", *QUIET, "--out", str(sd / f"seal_{l}.npz")))
              for l in LAYERS])

    cands = build_candidates(sd) if not late() else []
    order = sorted(cands, key=lambda n: (abs(float(n.split("_a")[1]) - 1.0), n))   # alpha 1 first, then the rest
    if cands and not late():
        progress(f"Generating {len(order)} steering candidates on dev128, paired with the starting model")
        jobs([(f"gen_{m}_dev128", wsp_case("run", CASE, m, "dev128", "--n", "2")) for m in order[: 2 * len(gpus())]],
             critical=False)
        best, res = pick_best(order)
        notes("dev128 (2 samples), paired gains overall and per source:\n```\n" + json.dumps(res, indent=1) + "\n```")
        best = if_guard(res, a.task)
        if best:
            keep_best(best, res[best]["all"]["gain"], "early result, passed the instruction-following check", a.task)

    b = best_so_far()
    if b and not late():
        progress(f"Refining {b['edit']}: dose and the two best layers together")
        layer, alpha = int(b["edit"][4:].split("_a")[0]), float(b["edit"].split("_a")[1])
        variants = []
        for f in (0.7, 1.4):
            name = f"SEAL{layer}_a{round(alpha * f, 2):g}"
            if not (wcase.cdir(CASE) / "edits" / f"{name}.json").exists():
                one(f"build_{name}", seal("build", "--case", CASE, "--data", str(sd / f"seal_{layer}.npz"),
                                          "--layer", str(layer), "--alpha", str(round(alpha * f, 2)), "--name", name))
            variants.append(name)
        res = compare(order, "dev128")
        others = sorted((m for m in res if m != b["edit"] and int(m[4:].split("_a")[0]) != layer and res[m]["all"]),
                        key=lambda m: -res[m]["all"]["gain"])
        if others:
            variants.append(merge(b["edit"], others[0], f"{b['edit']}+{others[0]}"))
        jobs([(f"gen_{m}_dev128", wsp_case("run", CASE, m, "dev128", "--n", "2")) for m in variants], critical=False)
        nb, res = pick_best([b["edit"], *variants])
        notes("Refinement, dev128 paired:\n```\n" + json.dumps({m: res[m] for m in res if m in variants}, indent=1)
              + "\n```")
        if nb and nb != b["edit"] and res[nb]["all"]["gain"] > b["dev128_gain"] + 1.0:
            ok = if_guard({nb: res[nb]}, a.task, top=1)
            if ok:
                keep_best(nb, res[nb]["all"]["gain"], "better variant, passed the instruction-following check", a.task)

    b = best_so_far()
    if b and time_left() > 75 * 60:
        progress(f"Confirming {b['edit']} on holdout256 (never used for any choice)")
        subprocess.run(wsp_case("freeze", CASE, b["edit"]), capture_output=True)
        jobs([(f"gen_{b['edit']}_holdout256", wsp_case("run", CASE, b["edit"], "holdout256"))], critical=False)
        notes("HOLDOUT256 (1 sample), the frozen edit:\n```\n" + json.dumps(compare([b["edit"]], "holdout256"), indent=1)
              + "\n```")
    if b:
        upload(b["edit"], a.task, "final, with the full report")
        progress(f"Uploaded {b['edit']}: {b['dev128_gain']:+.1f} points on dev128. Finished.")
    else:
        progress("No candidate beat the starting model on dev128 without a loss on some source; nothing uploaded.")
        log("no edit qualified")


def status(_a):
    ev = STATE / "events.log"
    print(ev.read_text()[-4000:] if ev.exists() else "not started")
    print("best:", json.dumps(best_so_far()), "| minutes left:", int(time_left() // 60))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--parent", help="the parent edit (parent/edit/edit.json of the task bundle); none: the base")
    p.add_argument("--task", help="task.json; without it nothing is uploaded (rehearsal)")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    {"run": run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
