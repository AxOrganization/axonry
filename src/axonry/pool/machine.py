"""What every GPU machine has in common, rented (pool/vast.py) or owned (pool/owned.py): ssh access as root (the
orchestrator) or as the run's agent, the setup bundle, the resources a run's agent is told about.

A machine is a container of the pinned image set up by setup.sh: /srv/axonry (the verified base, the competition repo,
evaluation jobs) belongs to root, the agent logs in as the unprivileged user `agent`. A machine behind a jump host
(an owned host's container, whose sshd listens on the host's loopback) is reached through that host's sshd.
"""
import gzip
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = "/srv/axonry"          # the machine's layout (setup.sh)
EVAL = "eval"                 # evaluation machines' purposes: eval, eval-2, ...; a run machine's is its submission id


def eval_names(n: int) -> list:
    """The purposes of n evaluation machines, the first one used first."""
    return [EVAL] + [f"{EVAL}-{i}" for i in range(2, n + 1)]


def is_eval(purpose: str) -> bool:
    return purpose == EVAL or purpose.startswith(f"{EVAL}-")
HERE = Path(__file__).parent
# IFEval's sentence tokenizer (nltk punkt_tab; sha256 as nltk's data index lists it). Fetched here and shipped in the
# setup bundle: the downloader's source is unreachable from some machines, and a grader's data must not change.
NLTK_PUNKT = ("e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
              ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
               "https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip"))


class MachineError(RuntimeError):
    pass


def _ed25519(known_hosts: str) -> str:
    for line in Path(known_hosts).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "ssh-ed25519":
            return f"{parts[1]} {parts[2]}"
    raise MachineError(f"no ed25519 host key recorded in {known_hosts}")


@dataclass
class Jump:
    """The host whose sshd forwards to the machine (only to it: the host's sshd permits nothing else)."""
    host: str
    port: int
    user: str
    known_hosts: str
    proxy_command: str | None = None

    def proxy(self, key=None) -> str:
        """A ProxyCommand reaching the target (%h:%p, as the jump host sees it) through this host."""
        args = ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-p", str(self.port),
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "StrictHostKeyChecking=accept-new"]
        if self.proxy_command:   # its own placeholders mean the jump host, not the target
            args += ["-o", "ProxyCommand=" + self.proxy_command.replace("%h", self.host).replace("%p", str(self.port))]
        if key:
            args += ["-i", key]
        return " ".join(shlex.quote(a) for a in args) + f" -W %h:%p {self.user}@{self.host}"

    def host_key(self) -> str:
        return _ed25519(self.known_hosts)


@dataclass
class Machine:
    """ssh access to a ready machine, as root (the orchestrator) or as the agent."""
    name: str
    ip: str
    port: int
    known_hosts: str
    proxy_command: str | None = None
    user: str = "root"
    jump: Jump | None = None

    @property
    def ssh(self):
        return f"{self.user}@{self.ip}"

    def ssh_args(self, key=None):
        args = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-p", str(self.port),
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "StrictHostKeyChecking=accept-new"]
        proxy = self.jump.proxy(key) if self.jump else self.proxy_command
        if proxy:
            args += ["-o", f"ProxyCommand={proxy}"]
        return args + (["-i", key] if key else [])

    def scp_args(self, key=None):
        return ["-P" if a == "-p" else a for a in self.ssh_args(key)]

    def host_key(self) -> str:
        """The machine's sshd key as first seen (the agent's known_hosts pins it)."""
        return _ed25519(self.known_hosts)


def ssh(m: Machine, key, command: str, timeout=900, attempts=3, binary=False):
    """Run a command on the machine -> its stdout (bytes with binary). ssh's own failure to connect (exit 255: a
    dropped or refused connection) is retried; every command we run is safe to repeat. A failing command is not."""
    for i in range(attempts):
        try:
            r = subprocess.run(["ssh", *m.ssh_args(key), m.ssh, command], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise MachineError(f"{m.name}: `{command[:120]}` timed out") from e
        if r.returncode != 255 or i == attempts - 1:
            break
        time.sleep(5 * (i + 1))
    if r.returncode != 0:
        raise MachineError(f"{m.name}: `{command[:120]}` failed: {r.stderr.decode(errors='replace').strip()[-500:]}")
    return r.stdout if binary else r.stdout.decode(errors="replace")   # logs may hold torn multibyte characters


def sample(m: Machine, key) -> dict:
    """A run's machine at a glance: each GPU's use and the agent's one-line status (the first line of its PROGRESS.md,
    at most 20 words; None before it writes one), both public; done: the agent ran axonry-done."""
    out = ssh(m, key, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits;"
              " echo ---; test -f /workspace/work/DONE && echo done; echo ---;"
              " head -c 2000 /workspace/work/PROGRESS.md 2>/dev/null || true", timeout=20, attempts=1)
    gpus_out, _, rest = out.partition("---\n")
    flag, _, progress = rest.partition("---\n")
    gpus = [dict(zip(("util_pct", "mem_used_mib", "mem_total_mib"), map(int, line.split(","))))
            for line in gpus_out.splitlines() if line.strip()]
    line = next((x for x in progress.splitlines() if x.strip()), "")
    line = re.sub(r"^(status:)\s*", "", line.strip().lstrip("#*->` ").strip(), flags=re.I)
    return {"gpus": gpus, "note": " ".join(line.split()[:20])[:200] or None, "done": flag.strip() == "done"}


def scp(m: Machine, key, srcs, dst: str, timeout=1800, attempts=3):
    """Copy local files or directories (one path or a list) to dst on the machine; retried, it is repeatable."""
    srcs = [str(x) for x in (srcs if isinstance(srcs, (list, tuple)) else [srcs])]
    for i in range(attempts):
        try:
            r = subprocess.run(["scp", "-q", "-r", *m.scp_args(key), *srcs, f"{m.ssh}:{dst}"], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise MachineError(f"{m.name}: scp timed out") from e
        if r.returncode == 0:
            return
        time.sleep(5 * (i + 1))
    raise MachineError(f"{m.name}: scp failed: {r.stderr.strip()[-500:]}")


def keypair(comment: str):
    """-> (private key, public key line): ed25519, no passphrase, written only to a temporary directory."""
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(p)], check=True)
        return p.read_text(), p.with_suffix(".pub").read_text().strip()


def punkt_tab(cache: Path) -> bytes:
    """nltk's punkt_tab.zip, sha256-checked (cached)."""
    import urllib.request  # noqa: PLC0415
    sha, urls = NLTK_PUNKT
    if cache.exists() and hashlib.sha256(cache.read_bytes()).hexdigest() == sha:
        return cache.read_bytes()
    for url in urls:
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() == sha:
            cache.write_bytes(data)
            return data
    raise MachineError("nltk punkt_tab: no source served the pinned file")


def setup_bundle(project: Path, competition: Path, base_docs: list, bench_docs: list, dest: Path) -> Path:
    """What setup.sh installs: our package (as the working tree has it), the competition repo at its checked-out
    commit (the orchestrator refuses to start unless that is the rules' pin), the rules' base definitions, its
    benchmarks (key, task, frozen manifest: what an owned evaluation host checks its items against) and nltk's
    tokenizer data."""
    from ..evaluation import benchdata  # noqa: PLC0415
    from ..registry import base_dirname  # noqa: PLC0415
    dest.parent.mkdir(parents=True, exist_ok=True)
    wsp = subprocess.run(["git", "-C", str(competition), "archive", "--format=tar", "HEAD"], capture_output=True,
                         check=True).stdout
    def plain(ti):   # no build time, owner or mode bits from this checkout: the same content gives the same bundle
        if "__pycache__" in ti.name:
            return None
        ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
        return ti

    with gzip.GzipFile(dest, "wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        def add(name, data, mode=0o644):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), mode
            tf.addfile(ti, io.BytesIO(data))
        for f in ("setup.sh", "stage_base.py"):
            add(f, (HERE / f).read_bytes(), 0o755)
        pkg = io.BytesIO()
        with gzip.GzipFile(fileobj=pkg, mode="wb", mtime=0) as pz, tarfile.open(fileobj=pz, mode="w") as p:
            for rel in ("pyproject.toml", "README.md", "src"):
                p.add(project / rel, arcname=f"axonry/{rel}", filter=plain)
        add("axonry.tar.gz", pkg.getvalue())
        add("wsp.tar", wsp)
        for doc in base_docs:
            add(f"bases/{base_dirname(doc)}.json", json.dumps(doc).encode())
        add("benchmarks.json", json.dumps(benchdata.spec(bench_docs)).encode())
        add("punkt_tab.zip", punkt_tab(dest.parent / "punkt_tab.zip"))
    return dest


def give_to_agent(m: Machine, key, public_key: str, deadline: int, run=ssh):
    """The run's login key for the agent user, and its deadline (axonry-time)."""
    q = shlex.quote
    run(m, key, f"install -d -m 700 -o agent -g agent ~agent/.ssh && echo {q(public_key)} > "
                "~agent/.ssh/authorized_keys && chown agent:agent ~agent/.ssh/authorized_keys && "
                f"chmod 600 ~agent/.ssh/authorized_keys && echo {int(deadline)} > /etc/axonry-deadline", timeout=120)


LIMITS = r'''python3 - <<'P'
import os
def first(*paths):
    for p in paths:
        try:
            return open(p).read().split()
        except OSError:
            pass
cpus = len(os.sched_getaffinity(0))
q = first("/sys/fs/cgroup/cpu.max") or (first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or ["-1"]) + \
    (first("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or ["100000"])
if q[0] not in ("max", "-1"):
    cpus = min(cpus, int(q[0]) // int(q[1]))
mem = int(open("/proc/meminfo").read().split()[1]) * 1024
lim = first("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes") or ["max"]
if lim[0] != "max":
    mem = min(mem, int(lim[0]))
print("CPUS", cpus)
print("MEM", mem >> 30)
P'''


def probe(m: Machine, key, network: str, run=ssh) -> dict:
    """What the run's agent gets, as seen on the machine (its container's limits; BRIEF.md lists it)."""
    out = run(m, key, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; " + LIMITS + "\n"
                      "echo DISK $(df -BG --output=avail /workspace/work | tail -1 | tr -dc 0-9)", timeout=120)
    kv = {line.split()[0]: line.split()[1] for line in out.splitlines() if line.split()[:1] in (["CPUS"], ["MEM"],
                                                                                               ["DISK"])}
    return {"gpus": [{"name": n.strip(), "memory_gib": round(int(mem) / 1024)} for n, mem in
                     (line.split(",") for line in out.splitlines() if "," in line)],
            "cpus": int(kv["CPUS"]), "memory": f"{kv['MEM']}g", "disk_gib": int(kv["DISK"]), "network": network,
            "base_path": "/workspace/base", "work_path": "/workspace/work",
            "competition_repo": "/workspace/WeightSpaceProgramming"}
