"""GPU hosts we own ([[machines.owned]] in the configuration), beside the machines rented on vast.ai.

A host has one role. An evaluation host runs one container on all its GPUs (the evaluation machine); a run host has
slots of a run's GPUs each, and every run gets a fresh container in a free slot, removed when the run ends. The
containers run the pinned image set up by setup.sh, exactly as a vast instance: the same layout, the same checks,
the same tools. pool/host.sh does the host's side (over ssh as root, on the host's ssh port):

  init   once per version of host.sh: Docker and the NVIDIA runtime, the firewall (containers open no connection to
         the host, nor to each other), the jump user, sshd, the pinned image. A host whose GPUs are busy with other
         processes fails it: freeing them is the operator's call.
  bake   once per setup bundle, on run hosts: the image + setup.sh committed as axonry-run:<bundle sha>, so a run's
         container holds everything it needs (no download, no setup by the agent) and is up in seconds.
  eval   once per setup bundle, on the evaluation host: its container set up, with the rules' benchmark items
         built by the orchestrator and checked against their manifests (evaluation/benchdata.py; never on a run
         host).

Containers listen on the host's loopback only (eval 2222, run slot n 2300+n): the orchestrator reaches them through
the host's sshd as root, a run's agent as the jump user, whose key may forward to its own container's port only.

Machine rows (the table shared with vast.ai; provider_id "owned:<host>:<slot>:<random>"):
  booting  the host steps above (detached on the host, polled each tick), then the container
  ready    the container is up
  gone     released: a run's container is removed; the evaluation container stays for the next use
"""
import hashlib
import logging
import shlex
import time
import uuid
from pathlib import Path

from .machine import HERE, ROOT, Jump, Machine, MachineError, give_to_agent, is_eval, probe, rsync, scp, ssh

log = logging.getLogger("axonry.owned")
OWNED = "owned:"
EVAL_SLOT = "eval"
EVAL_PORT = 2222
RUN_PORT0 = 2300              # run slot n listens on 2300 + n (host.sh permits slots 1..8)
JUMP_USER = "axjump"
HOST_DIR = "/srv/axonry/host"


class SetupFailed(MachineError):
    pass


def _sha12(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class OwnedPool:
    def __init__(self, hosts: list, db, *, image: str, ssh_key: str, bundle, items, state_dir: Path,
                 proxy_command=None, setup_minutes=180, clock=time.time, remote=None):
        """hosts: [{"name", "address", "port" (ssh, default 22), "role": "run" | "eval", "china": bool}]. bundle, items: () -> path of the setup
        bundle, of the benchmark items (built when an evaluation host first needs them). setup_minutes: a first init (Docker, a ~25 GB image) can take long; it continues on the host when a
        machine row gives up. remote: (ssh, scp, rsync) for tests."""
        self.hosts = {h["name"]: h for h in hosts}
        self.db, self.image, self.key, self.bundle, self.items = db, image, str(Path(ssh_key).expanduser()), bundle, items
        self.dir, self.proxy, self.setup_s, self.clock = Path(state_dir), proxy_command, setup_minutes * 60, clock
        self.ssh, self.scp, self.rsync = remote or (ssh, scp, rsync)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.version = _sha12(HERE / "host.sh")
        self._pushed, self._gpus = {}, {}

    # ---- the hosts
    def _host(self, name) -> Machine:
        h = self.hosts[name]
        return Machine(name, h["address"], int(h.get("port", 22)), str(self.dir / f"host-{name}.known_hosts"), self.proxy)

    def _jump(self, name, user) -> Jump:
        return Jump(self.hosts[name]["address"], int(self.hosts[name].get("port", 22)), user,
                    str(self.dir / f"host-{name}.known_hosts"),
                    self.proxy if user == "root" else None)

    def _hsh(self, name, args: str, timeout=300) -> str:
        return self.ssh(self._host(name), self.key, f"{HOST_DIR}/host.sh {args}", timeout=timeout).strip()

    def _china(self, name) -> str:
        return "1" if self.hosts[name].get("china") else "0"

    def _gpus_of(self, name) -> list:
        if name not in self._gpus:
            out = self.ssh(self._host(name), self.key, "nvidia-smi --query-gpu=name --format=csv,noheader", timeout=60)
            self._gpus[name] = [g.strip() for g in out.splitlines() if g.strip()]
        return self._gpus[name]

    def _push(self, name) -> str:
        """host.sh, the container entrypoint, our key for the containers' root, the setup bundle (on the host as
        setup-<sha>.tar) and, to an evaluation host, the benchmark items; once per process and bundle. -> the bundle's
        sha."""
        bundle = Path(self.bundle())
        sha = _sha12(bundle)
        remote = f"{HOST_DIR}/setup-{sha}.tar"
        if self._pushed.get(name) != sha:
            h = self._host(name)
            keys = self.dir / "authorized_keys"
            keys.write_text(Path(self.key + ".pub").read_text())
            self.ssh(h, self.key, f"mkdir -p {HOST_DIR}/init", timeout=60)
            self.scp(h, self.key, [HERE / "host.sh"], f"{HOST_DIR}/host.sh")
            self.scp(h, self.key, [HERE / "container_init.sh", keys], f"{HOST_DIR}/init/")
            self._upload(h, bundle, remote, basis=f"{HOST_DIR}/setup-*.tar")
            self.ssh(h, self.key, f"find {HOST_DIR} -maxdepth 1 -name 'setup-*' ! -name setup-{sha}.tar -delete",
                     timeout=60)
            if self.hosts[name]["role"] == "eval":
                self._upload(h, Path(self.items()), f"{HOST_DIR}/items.tar")
            self.ssh(h, self.key, f"chmod 755 {HOST_DIR}/host.sh {HOST_DIR}/init/container_init.sh", timeout=60)
            self._pushed[name] = sha
        return sha

    def _upload(self, h, path: Path, remote: str, basis=None):
        """Unless the host already holds this exact file (a link to a far host can take many minutes): scp, or with
        basis (a glob of earlier versions on the host) an rsync delta against the newest of them."""
        want = hashlib.sha256(path.read_bytes()).hexdigest()
        held = lambda: (self.ssh(h, self.key, f"sha256sum {shlex.quote(remote)} 2>/dev/null | cut -c1-64; true",  # noqa: E731
                                 timeout=120) or "").strip()
        if held() == want:
            return
        q = shlex.quote(remote)
        if not basis or "rsync" not in (self.ssh(h, self.key, "command -v rsync; true", timeout=60) or ""):
            self.scp(h, self.key, [path], remote)       # nothing to build on, or no rsync there: the whole file
        else:
            self.ssh(h, self.key, f"[ -f {q} ] || {{ old=$(ls -t {basis} 2>/dev/null | head -1); "
                                  f"[ -z \"$old\" ] || cp \"$old\" {q}; }}; true", timeout=120)
            self.rsync(h, self.key, path, remote)
        if held() != want:
            raise MachineError(f"{h.name}: {remote} does not match after the transfer")

    def _step(self, name, step, args) -> bool:
        """A detached host step: True once done; raises SetupFailed with its log when it failed (the next call
        starts it again)."""
        out = self._hsh(name, f"start {step} {args}")
        if out.startswith("failed"):
            raise SetupFailed(f"{name}: {step} failed: {out[6:].strip()[-600:]}")
        return out == "ok"

    # ---- rows
    @staticmethod
    def _where(row):
        _, host, slot, _ = row["provider_id"].split(":")
        return host, slot

    @staticmethod
    def _port(slot) -> int:
        return EVAL_PORT if slot == EVAL_SLOT else RUN_PORT0 + int(slot)

    @staticmethod
    def _container(slot) -> str:
        return "axonry-eval" if slot == EVAL_SLOT else f"axonry-run{slot}"

    def get(self, purpose) -> dict | None:
        return next((m for m in self.db.machines() if m["name"] == purpose), None)

    def machine(self, row) -> Machine:
        host, slot = self._where(row)
        return Machine(row["name"], "127.0.0.1", self._port(slot), str(self.dir / f"{row['provider_id']}.known_hosts"),
                       jump=self._jump(host, "root"))

    def rent(self, purpose: str, gpu_name: str, gpus: int) -> bool:
        """A free slot on a host of that GPU model (runs: a run host; evaluation: the evaluation host, if it has
        exactly `gpus` GPUs). -> False when there is none."""
        taken = {self._where(r) for r in self.db.machines() if r["provider_id"].startswith(OWNED)}
        for name, h in self.hosts.items():
            try:
                inv = self._gpus_of(name)
            except MachineError as e:
                log.warning("owned host %s unreachable: %s", name, str(e)[:200])
                continue
            if not inv or not all(gpu_name in g for g in inv):
                continue
            if is_eval(purpose):
                if h["role"] != "eval" or (name, EVAL_SLOT) in taken or len(inv) != gpus:
                    continue
                slot = EVAL_SLOT
            else:
                if h["role"] != "run":
                    continue
                free = [str(n) for n in range(1, len(inv) // gpus + 1) if (name, str(n)) not in taken]
                if not free:
                    continue
                slot = free[0]
            self.db.add_machine(purpose, f"{OWNED}{name}:{slot}:{uuid.uuid4().hex[:12]}", name, inv[0], gpus, 0.0,
                                self.clock())
            log.info("%s: owned host %s, slot %s", purpose, name, slot)
            return True
        return False

    def advance(self, purpose) -> str:
        """booting -> ready, or "failed" (the row is gone). Connection problems are retried on the next call."""
        row = self.get(purpose)
        if row is None:
            return "gone"
        if row["state"] == "ready":
            return "ready"
        host, slot = self._where(row)
        try:
            if self.clock() - row["created_at"] > self.setup_s:
                raise SetupFailed(f"not ready within {self.setup_s // 60} min")
            sha = self._push(host)
            bundle = f"{HOST_DIR}/setup-{sha}.tar"
            if not self._step(host, f"init-{self.version}", f"init {shlex.quote(self.image)} {self._china(host)}"):
                return "booting"
            if slot == EVAL_SLOT:
                if not self._step(host, f"eval-{sha}", f"eval {bundle} {HOST_DIR}/items.tar {shlex.quote(self.image)} "
                                                        f"{self._china(host)}"):
                    return "booting"
                if self._hsh(host, "state axonry-eval") != f"true {sha}":    # removed or replaced since: again
                    self.ssh(self._host(host), self.key, f"rm -f {HOST_DIR}/eval-{sha}.ok", timeout=60)
                    return "booting"
            else:
                if not self._step(host, f"bake-{sha}", f"bake {bundle} {shlex.quote(self.image)} {self._china(host)}"):
                    return "booting"
                if self._hsh(host, f"state axonry-run{slot}") != f"true {row['provider_id']}":
                    first = (int(slot) - 1) * row["gpus"]
                    gpus = ",".join(str(i) for i in range(first, first + row["gpus"]))
                    self._hsh(host, f"run {slot} {gpus} axonry-run:{sha} {row['provider_id']}", timeout=900)
            self.ssh(self.machine(row), self.key, f"test -f {ROOT}/READY", timeout=60, attempts=1)
            self.db.set_machine(purpose, state="ready", ready_at=self.clock(), ssh_host="127.0.0.1",
                                ssh_port=self._port(slot))
            log.info("machine %s ready (owned host %s, slot %s)", purpose, host, slot)
            return "ready"
        except SetupFailed as e:
            log.warning("machine %s (owned host %s): %s", purpose, host, e)
            self.release(purpose, f"failed: {e}"[:500])
            return "failed"
        except MachineError as e:
            log.info("machine %s (owned host %s): %s (retrying)", purpose, host, str(e)[:200])
            return row["state"]

    def release(self, purpose, note="released"):
        row = self.get(purpose)
        if row is None:
            return
        host, slot = self._where(row)
        if slot != EVAL_SLOT:      # the evaluation container stays (set up, idle) for the next evaluation
            try:
                self._hsh(host, f"stop axonry-run{slot}", timeout=120)
            except MachineError as e:   # the slot's next run replaces it
                log.warning("owned host %s: removing run container %s: %s", host, slot, str(e)[:200])
        self.db.set_machine(purpose, state="gone", ended_at=self.clock(), note=note)
        Path(self.dir / f"{row['provider_id']}.known_hosts").unlink(missing_ok=True)
        log.info("machine %s released (owned host %s, slot %s): %s", purpose, host, slot, note)

    def hand_over(self, purpose, public_key: str, deadline: int) -> Machine:
        """The run's agent: its key in the container, and on the host's jump user for this container's port only.
        -> the machine as the agent reaches it."""
        row = self.get(purpose)
        host, slot = self._where(row)
        m = self.machine(row)
        give_to_agent(m, self.key, public_key, deadline, self.ssh)
        self._hsh(host, f"jump {self._port(slot)} {shlex.quote(public_key)}", timeout=60)
        return Machine(m.name, "127.0.0.1", m.port, m.known_hosts, None, "agent", self._jump(host, JUMP_USER))

    def probe(self, purpose) -> dict:
        row = self.get(purpose)
        host, _ = self._where(row)
        net = ("limited (mainland China: Hugging Face, GitHub and PyPI may be unreachable; everything the run needs "
               "is installed)") if self.hosts[host].get("china") else "open"
        return probe(self.machine(row), self.key, net, self.ssh)

    def vanished(self, purpose, row=None) -> bool:
        """A ready machine whose container no longer runs (host rebooted, removed by hand); an unreachable host is
        not (yet) a vanished machine."""
        row = row or self.get(purpose)
        if row is None or row["state"] != "ready":
            return False
        host, slot = self._where(row)
        try:
            state = self._hsh(host, f"state {self._container(slot)}", timeout=60)
        except MachineError:
            return False
        return not (state.startswith("true ") if slot == EVAL_SLOT else state == f"true {row['provider_id']}")
