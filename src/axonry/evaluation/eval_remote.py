"""Evaluation of one model on every benchmark of the rules, on the rented evaluation machine (axonry installed by
pool/setup.sh; the competition toolkit at /srv/axonry/repos/WeightSpaceProgramming, on PYTHONPATH).

One vLLM server spans all evaluation GPUs (data parallel). Every (item, sample) of every benchmark, in benchmark
order, is one request of a single stream kept at `concurrency` in flight: a slot freed by any answer takes the next
request, so the GPUs stay full and no benchmark waits for another's long tail. Prompts, per-request seeds
(wsp.common.stable_seed: item id, sample, salt) and grading are the toolkit's own, so a model is paired with every
other model of the same base item by item.

Each graded answer is appended to <out>/<key>.part.jsonl at once; a rerun skips what is there (and grades again what
is there, so every row is graded by this code). A benchmark's file becomes <out>/<key>.jsonl when it is complete.
Grading runs in the main thread only: the toolkit's math check (math_verify) times itself out with signal.alarm, which
fails in any other thread - the answer would be marked wrong. Exit codes: 0 done; 3 the frozen item set does not match (ours, never
the model's); anything else: the run failed.

  python3 -m axonry.evaluation.eval_remote --model DIR --profile BASE_ID --salt S --out DIR --gpus N
         --concurrency C --benchmarks '[{"key", "task", "n", "limit", "manifest"}, ...]'
"""
import argparse
import json
import resource
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ITEMS_MISMATCH = 3


def requests_for(benchmarks, load, prompt_of, stable_seed, salt):
    """-> [(key, item, k, prompt, seed)] in benchmark order, then item order, then sample."""
    out = []
    for b in benchmarks:
        items = load(b["task"])
        for r in items[:b.get("limit") or None]:
            p = prompt_of(r)
            out += [(b["key"], r, k, p, stable_seed(r["id"], k, salt)) for k in range(b["n"])]
    return out


def read_done(part: Path) -> dict:
    """Rows already written (a torn last line from a crash is dropped and the file rewritten clean)."""
    rows = []
    if part.exists():
        for line in part.read_text().split("\n"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {(str(r["id"]), r["k"]): r for r in rows}


def graded(grade, r, text, finish) -> dict:
    try:
        return grade(r, text, finish)
    except Exception as e:  # noqa: BLE001 - an answer the grader cannot read is wrong, never a failed run
        return {"pred": None, "correct": False, "cut": finish == "length", "kind": "grade_error",
                "grade_error": f"{type(e).__name__}: {e}"[:300]}


def regrade(row, r, grade) -> dict:
    """A stored row graded again by this code (its answer text is kept)."""
    keep = {k: row[k] for k in ("id", "k", "source", "gold", "n_tokens", "finish")}
    return {**keep, **graded(grade, r, row["text"], row["finish"]), "text": row["text"]}


def run_stream(reqs, send, grade, out: Path, concurrency: int):
    """Keep `concurrency` requests in flight over the whole stream; write each graded row as it lands; finalize every
    benchmark whose rows are all there. send(prompt, seed) -> (text, finish_reason, n_tokens)."""
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(q[0] for q in reqs))
    final = {key for key in keys if (out / f"{key}.jsonl").exists()}       # finished by an earlier run
    done = {key: read_done(out / f"{key}.part.jsonl") for key in keys if key not in final}
    items = {(q[0], str(q[1]["id"])): q[1] for q in reqs}
    for key in keys:                                                       # rows of an earlier run: graded again
        path = out / (f"{key}.jsonl" if key in final else f"{key}.part.jsonl")
        if path.exists():
            rows = [regrade(r, items[(key, str(r["id"]))], grade) for r in read_done(path).values()]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            if key in done:
                done[key] = {(str(r["id"]), r["k"]): r for r in rows}
    todo = [q for q in reqs if q[0] not in final and (str(q[1]["id"]), q[2]) not in done[q[0]]]
    files = {key: open(out / f"{key}.part.jsonl", "a") for key in done}

    def one(q):                         # in a worker thread: the answer only
        return q, send(q[3], q[4])

    def write(q, answer):               # in the main thread: grading (see the module doc)
        key, r, k = q[0], q[1], q[2]
        text, finish, ntok = answer
        row = {"id": r["id"], "k": k, "source": r.get("source"), "gold": r.get("answer"), "n_tokens": ntok,
               "finish": finish, **graded(grade, r, text, finish), "text": text}
        files[key].write(json.dumps(row) + "\n")
        files[key].flush()
        return ntok

    t0, ntok, it = time.time(), 0, iter(todo)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        running = {pool.submit(one, q) for q in (next(it, None) for _ in range(concurrency)) if q is not None}
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                ntok += write(*f.result())    # a failed request fails the run (rerun resumes)
                q = next(it, None)
                if q is not None:
                    running.add(pool.submit(one, q))
    for f in files.values():
        f.close()
    want = {}
    for q in reqs:
        want[q[0]] = want.get(q[0], 0) + 1
    for key in done:
        if len(read_done(out / f"{key}.part.jsonl")) == want[key]:
            (out / f"{key}.part.jsonl").replace(out / f"{key}.jsonl")
    dt = time.time() - t0
    print(f"[eval] {len(todo)} answers ({len(reqs) - len(todo)} resumed), {ntok} tokens in {dt:.0f}s "
          f"({ntok / max(dt, 1):.0f} tok/s)", flush=True)


def http_sender(port, model, sampling, attempts=3):
    url = f"http://127.0.0.1:{port}/v1/completions"

    def send(prompt, seed):
        body = json.dumps({"model": model, "prompt": prompt, "seed": seed, **sampling}).encode()
        for i in range(attempts):
            try:
                req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=6 * 3600) as resp:
                    d = json.loads(resp.read())
                c = d["choices"][0]
                return c["text"], c["finish_reason"], d["usage"]["completion_tokens"]
            except Exception:  # noqa: BLE001 - a server hiccup: retried, then the run fails (and resumes)
                if i == attempts - 1:
                    raise
                time.sleep(5)
    return send


def start_server(model, gpus, max_len, port):
    from transformers import AutoConfig  # noqa: PLC0415
    argv = ["vllm", "serve", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--data-parallel-size", str(gpus), "--dtype", "bfloat16", "--max-model-len", str(max_len),
            "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--seed", "0"]
    if getattr(AutoConfig.from_pretrained(str(model)), "vision_config", None) is not None:
        argv += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    proc = subprocess.Popen(argv, stdout=sys.stderr, stderr=sys.stderr)
    for _ in range(3600):
        if proc.poll() is not None:
            raise SystemExit(f"vllm serve exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return proc
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(1)
    proc.kill()
    raise SystemExit("vllm serve did not come up")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)   # one socket per request in flight (gpus x concurrency)
    want = 1 << 20 if hard == resource.RLIM_INFINITY else min(hard, 1 << 20)
    if soft != resource.RLIM_INFINITY and soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    from transformers import AutoTokenizer  # noqa: PLC0415
    from wsp import data, profiles  # noqa: PLC0415
    from wsp.common import stable_seed  # noqa: PLC0415
    from wsp.gen import CONTEXT_MARGIN  # noqa: PLC0415
    from wsp.grade import score  # noqa: PLC0415

    benchmarks = json.loads(a.benchmarks)
    for b in benchmarks:
        try:
            got = data.manifest(b["task"])["sha256"]
        except FileNotFoundError as e:      # the items were not supplied to this machine: ours, never the model's
            print(f"{b['key']}: items missing: {e}", file=sys.stderr)
            return ITEMS_MISMATCH
            print(f"{b['key']}: item set {got[:12]} != frozen {b['manifest'][:12]}", file=sys.stderr)
            return ITEMS_MISMATCH
    prof = profiles.get(a.profile)
    tok = AutoTokenizer.from_pretrained(a.model)

    def prompt_of(r):
        return tok.apply_chat_template([{"role": "user", "content": data.user_message(r)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=prof["thinking"])

    reqs = requests_for(benchmarks, data.load, prompt_of, stable_seed, a.salt)
    sampling = prof["sampling"]
    server = start_server(a.model, a.gpus, sampling["max_tokens"] + CONTEXT_MARGIN, a.port)
    try:
        run_stream(reqs, http_sender(a.port, str(a.model), sampling),
                   lambda r, text, finish: score(r, text, finish, prof["thinking"]), a.out, a.concurrency)
    finally:
        server.terminate()
        server.wait(timeout=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
