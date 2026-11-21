"""Work on the owner-only evaluation machine (an owned host's or a rented one: pool/): admission build and
benchmark runs with the competition repo and the verified base. No miner code runs there: a submission is data (an
edit spec), applied by our admission code.

Machine layout (pool/setup.sh, root /srv/axonry): models/<org>__<name>@<revision>/ (verified),
repos/WeightSpaceProgramming/ (the pinned commit), wsp_work/ (WSP_WORK: the benchmark items, supplied at setup;
its norm cache), eval/<model_id>/ (spec, built model, outputs; base -> the verified base). Jobs run in eval/<model_id>/.
"""
import json
import shlex
import time
from pathlib import Path

from ..pool import machine as machines


class ExecError(RuntimeError):
    """Our side: the host is unreachable, a transfer failed, the frozen data does not match. Never the submission's
    fault: retried with backoff, never counted against a submission."""


class JobFailed(ExecError):
    """A job ran and exited non-zero (or never finished). When it is a candidate's own job, the submission may be
    the cause, so it is counted."""


class SshExecutor:
    def __init__(self, machine, ssh_key: str | None = None, gpus: int = 8, concurrency_per_gpu: int = 256):
        """machine: (purpose) -> pool.machine.Machine of that evaluation machine (raises ExecError while it is not
        ready); every evaluation uses all its GPUs (one vLLM server, data parallel). Every call names the evaluation
        machine it works on (`on`)."""
        self.machine, self.key, self.root, self.gpus = machine, ssh_key, machines.ROOT, gpus
        self.concurrency = concurrency_per_gpu * gpus

    def ready(self, on) -> bool:
        try:
            self.machine(on)
            return True
        except ExecError:
            return False

    def _ssh(self, on, remote_cmd: str, timeout=600, binary=False):
        try:
            return machines.ssh(self.machine(on), self.key, remote_cmd, timeout=timeout, binary=binary)
        except machines.MachineError as e:
            raise ExecError(f"eval host {on}: {e}") from e

    def fetch_file(self, on, path: str, attempts=3) -> bytes:
        """A file from the evaluation host, verified: sha256 computed there, gzip transfer, checked here (a long
        transfer through a proxy can be cut silently)."""
        import gzip  # noqa: PLC0415
        import hashlib  # noqa: PLC0415
        want = self._ssh(on, f"sha256sum {shlex.quote(path)}").split()[0]
        for _ in range(attempts):
            try:
                data = gzip.decompress(self._ssh(on, f"gzip -c {shlex.quote(path)}", timeout=3600, binary=True))
            except (OSError, EOFError, ExecError):
                continue
            if hashlib.sha256(data).hexdigest() == want:
                return data
        raise ExecError(f"could not fetch {path} intact")

    def _command(self, model_id: str, base_dirname: str, argv: list[str]):
        r = self.root
        env = {"PYTHONPATH": f"{r}/repos/WeightSpaceProgramming", "WSP_MODELS_DIR": f"{r}/models",
               "NLTK_DATA": "/usr/share/nltk_data", "WSP_WORK": f"{r}/wsp_work",
               # vLLM's compile cache is kept: it depends on the architecture, not on the weights
               "VLLM_CACHE_ROOT": f"{r}/wsp_work/vllm_cache"}
        return ["env", *(f"{k}={v}" for k, v in env.items()), *argv], f"ln -sfn {r}/models/{base_dirname} {r}/eval/{model_id}/base"

    def _start(self, on, model_id: str, cmd: list, gpus: bool, setup: str = "true") -> str:
        """Start a command on the evaluation host detached from this ssh connection (a long generation must survive
        network hiccups and orchestrator restarts). GPU jobs hold the host-side lock of its GPUs (flock). The job
        writes out / err / exit files. -> the job's directory."""
        q = shlex.quote
        job = f"{self.root}/eval/{model_id}/jobs/{time.time_ns()}"
        run = " ".join(map(q, cmd))
        if gpus:
            run = f"flock {q(self.root + '/eval/gpus.lock')} {run}"
        body = (f"cd {q(self.root + '/eval/' + model_id)} && {setup} && {run} > {q(job)}/out 2> {q(job)}/err; "
                f"echo $? > {q(job)}/exit")
        # two statements: only the fully redirected nohup goes to the background, so ssh returns at once
        self._ssh(on, f"mkdir -p {q(job)}; nohup sh -c {q(body)} > /dev/null 2>&1 < /dev/null &", timeout=120)
        return job

    def _exit(self, on, job: str) -> str:
        """The job's exit code, "" while it runs."""
        return self._ssh(on, f"cat {shlex.quote(job)}/exit 2>/dev/null || true", timeout=60).strip()

    def _check(self, on, job: str, code: str, ok=("0",)):
        err = lambda: self._ssh(on, f"tail -c 800 {shlex.quote(job)}/err")  # noqa: E731
        if code == "3":             # eval_remote.ITEMS_MISMATCH: our frozen data, never the model's fault
            raise ExecError(f"job {job}: {err()}")
        if code not in ok:
            raise JobFailed(f"job {job} exited {code}: {err()}")

    def _job(self, on, model_id: str, cmd: list, gpus: bool, timeout: int, poll: int = 20, setup: str = "true",
             ok=("0",)) -> str:
        """A short job, waited for here: -> its stdout."""
        job = self._start(on, model_id, cmd, gpus, setup)
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(poll)
            try:
                code = self._exit(on, job)
            except ExecError:
                continue          # the host is unreachable for a moment; the job keeps running
            if code:
                self._check(on, job, code, ok)
                return self.fetch_file(on, f"{job}/out").decode()
        raise JobFailed(f"job {job} did not finish within {timeout}s")

    def push_dir(self, on, local: Path, model_id: str, sub: str):
        """local's files to eval/<model_id>/<sub>/ on the machine, by content: a file the machine already holds (a
        start's published edit placed in many builds, a retry after a cut transfer) is copied there, not sent again;
        each missing file is sent on its own and kept once complete (eval/_cache/<sha256>, a week)."""
        import hashlib  # noqa: PLC0415
        q, dst, cache = shlex.quote, f"{self.root}/eval/{model_id}/{sub}", f"{self.root}/eval/_cache"
        files = {p.relative_to(local).as_posix(): (p, hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(Path(local).rglob("*")) if p.is_file()}
        place = " && ".join(f"mkdir -p {q(str(Path(dst, rel).parent))} && "
                            f"{{ cp {cache}/{h} {q(f'{dst}/{rel}')} 2>/dev/null || echo {h}; }}"
                            for rel, (_, h) in files.items())
        prep = (f"mkdir -p {cache} && find {cache} -type f -mtime +7 -delete; rm -rf {q(dst)} && mkdir -p {q(dst)}"
                + (f" && {place}" if place else ""))
        missing = set(self._ssh(on, prep, timeout=300).split())
        try:
            for h in sorted(missing):                    # each content sent once, then placed wherever it goes
                p = next(p for p, h2 in files.values() if h2 == h)
                machines.scp(self.machine(on), self.key, [p], f"{cache}/{h}.part", timeout=900)
                place = " && ".join(f"cp {cache}/{h} {q(f'{dst}/{rel}')}" for rel, (_, h2) in files.items() if h2 == h)
                self._ssh(on, f"test \"$(sha256sum {cache}/{h}.part | cut -d' ' -f1)\" = {h} && "
                              f"mv {cache}/{h}.part {cache}/{h} && {place}", timeout=300)
        except machines.MachineError as e:
            raise ExecError(f"eval host {on}: {e}") from e


    def admit(self, on, sid: str, spec_dir: Path, base_doc: dict) -> dict:
        from ..registry import base_dirname  # noqa: PLC0415
        base_id = base_doc["id"]
        self.push_dir(on, spec_dir, sid, "edit")
        cmd, ln = self._command(sid, base_dirname(base_doc), [   # CPU only: runs alongside an evaluation
            "python3", "-m", "axonry.evaluation.admit_remote", "--spec-dir", "edit",
            "--base-dir", "base", "--base-id", base_id, "--build-dir", "model"])
        out = self._job(on, sid, cmd, gpus=False, timeout=3600, setup=ln,
                        ok=("0", "1"))     # 1: a verdict that does not pass (rejected under the rules), not a crash
        try:
            return json.loads(out[out.index("{"):])
        except ValueError:      # it crashed (exit 1) or printed no verdict: the candidate's own job failed
            raise JobFailed(f"admission produced no result: {out[-500:]}") from None

    def flatten(self, on, sid: str, base_doc: dict) -> tuple[dict, dict]:
        """A champion's chain (placed by admission on that machine) as one edit on the base, checked identical to its
        evaluated model there (axonry.evaluation.flatten_remote, CPU only) -> (its result, {file name: bytes} when ok)."""
        from ..registry import base_dirname  # noqa: PLC0415
        d = f"{self.root}/eval/{sid}"
        gone = self._ssh(on, f"cd {shlex.quote(d)} && for f in edit/edit.json model/config.json; do test -e $f || "
                             "echo $f; done", timeout=120).split()
        if gone:
            return {"ok": False, "problem": f"its evaluated model is no longer on {on} (no {', '.join(gone)})"}, {}
        cmd, ln = self._command(sid, base_dirname(base_doc), [
            "python3", "-m", "axonry.evaluation.flatten_remote", "--edit", "edit", "--model", "model",
            "--base-dir", "base", "--out", "flat"])
        out = self._job(on, sid, cmd, gpus=False, timeout=3600, setup=ln, ok=("0", "1"))
        try:
            res = json.loads(out[out.index("{"):])
        except ValueError:
            raise ExecError(f"flatten produced no result: {out[-500:]}") from None
        if not res.get("ok"):
            return res, {}
        names = self._ssh(on, f"ls {shlex.quote(d + '/flat')}").split()
        return res, {n: self.fetch_file(on, f"{d}/flat/{n}") for n in names}

    @staticmethod
    def _spec(bench_docs):
        return [{"key": f"{b['id']}@{b['version']}", "task": b["task"], "n": b["samples_per_item"],
                 "limit": b.get("limit"), "manifest": b["items_manifest_sha256"]} for b in bench_docs]

    def start(self, on, model_id: str, base_doc: dict, bench_docs: list, salt: str) -> str:
        """One model on the given benchmarks, all GPUs of that evaluation machine, one detached job
        (axonry.evaluation.eval_remote). model_id "base:<id>" runs the verified base itself; otherwise the built
        model of that submission. Partial output on the machine survives a failure and is resumed by the next job
        there. -> the job, for result()."""
        from ..registry import base_dirname  # noqa: PLC0415
        model_dir = "base" if model_id.startswith("base:") else "model"
        safe = model_id.replace(":", "_")
        cmd, ln = self._command(safe, base_dirname(base_doc), [
            "python3", "-m", "axonry.evaluation.eval_remote", "--model", model_dir, "--profile", base_doc["id"],
            "--salt", salt, "--out", f"out/{salt}", "--gpus", str(self.gpus),
            "--concurrency", str(self.concurrency), "--benchmarks", json.dumps(self._spec(bench_docs))])
        return self._start(on, safe, cmd, gpus=True, setup=ln)

    def result(self, on, job: str, model_id: str, bench_docs: list, salt: str) -> dict | None:
        """None while the job runs; -> {bench key: output JSON lines} once it finished; raises when it failed."""
        try:
            code = self._exit(on, job)
        except ExecError:
            return None           # unreachable for a moment: the job keeps running
        if not code:
            return None
        self._check(on, job, code)
        safe = model_id.replace(":", "_")
        return {b["key"]: self.fetch_file(on, f"{self.root}/eval/{safe}/out/{salt}/{b['key']}.jsonl")
                for b in self._spec(bench_docs)}

    def evaluate(self, on, model_id: str, base_doc: dict, bench_docs: list, salt: str, poll: int = 30) -> dict:
        """start() and wait for its result (a tool's one-off evaluation, e.g. calibration)."""
        job = self.start(on, model_id, base_doc, bench_docs, salt)
        while (out := self.result(on, job, model_id, bench_docs, salt)) is None:
            time.sleep(poll)
        return out
