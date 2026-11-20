"""GPU machines rented on vast.ai: one machine per run (its sandbox) and one for evaluation, rented when
needed and destroyed when done.

A machine is a plain vast instance (an unprivileged container: no docker inside, no firewall): the instance itself is
the run's sandbox. The agent logs in as the unprivileged user `agent`; root (the orchestrator's key) keeps the
verified base, the competition repo and the tools read-only for it. Instances of different runs are separate vast
containers (often the two halves of one 8-GPU host), so runs never share GPUs, memory, files or processes.

Lifecycle (table machines, one row per purpose: a submission id, or "eval"):
  booting     rented; waiting for the instance to run and its sshd to answer
  setting_up  setup bundle copied, setup.sh running (tools, competition repo, verified base, the agent user)
  ready       in use
  gone        destroyed (released, failed or vanished); a vast machine whose setup failed is not rented again
Setup time is ours: a run's clock starts when its machine is ready.
"""
import hashlib
import json
import logging
import os
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path

from .machine import ROOT, Machine, MachineError, give_to_agent, is_eval, probe, scp, ssh

UNREACHABLE_S = 10 * 60        # a container vast.ai reports running whose sshd never answers is given up after this

# on the machine: python3 -c FETCH <url> <dest> <sha256>: download, then refuse a file of another content
FETCH = """import hashlib, sys, time, urllib.request
url, dest, sha = sys.argv[1:4]
for attempt in range(5):
    try:
        urllib.request.urlretrieve(url, dest)
        break
    except OSError as e:
        print("download failed:", e, flush=True)
        time.sleep(10)
got = hashlib.sha256(open(dest, "rb").read()).hexdigest()
sys.exit(0 if got == sha else f"bundle sha256 {got} != {sha}")
"""

log = logging.getLogger("axonry.vast")
API = "https://console.vast.ai/api/v0"
KEY_GRACE = 15 * 60           # a running instance that still refuses our key after this never got it: a broken host


class VastError(MachineError):
    pass


class LowCredit(VastError):
    """The vast.ai account's credit is too low to rent: nothing failed, the queue waits for a top-up."""


class SetupFailed(VastError):
    """The machine will not become ready (setup.sh failed, or not ready in time): it is destroyed."""


class VastAPI:
    def __init__(self, api_key: str, base=API, timeout=60):
        self.key, self.base, self.timeout = api_key, base, timeout

    def call(self, method, path, body=None, query=None):
        url = self.base + path
        if query:
            url += "?" + "&".join(f"{k}={urllib.request.quote(json.dumps(v) if not isinstance(v, str) else v)}"
                                  for k, v in query.items())
        req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise VastError(f"{method} {path}: {e.code} {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise VastError(f"{method} {path}: {e}") from e

    def offers(self, gpu_name: str, gpus: int, min_reliability: float, disk_gb: int, min_down_mbps: int,
               exclude_countries=(), verified_only=False, min_cuda=0.0) -> list:
        """Rentable on-demand offers of exactly `gpus` GPUs of that model, cheapest first (only hosts vast has
        verified if verified_only). A slow link cannot pull the image in time, so offers below min_down_mbps are left
        out; so are hosts in countries whose route to PyPI, Hugging Face and R2 is unreliable (exclude_countries, ISO
        codes), and hosts whose driver cannot run the image's CUDA natively (min_cuda: consumer GPUs have no forward
        compatibility); never hosts vast has deverified. A host that turns out bad fails its setup and is never rented
        again."""
        q = {"gpu_name": {"eq": gpu_name}, "num_gpus": {"eq": gpus}, "rentable": {"eq": True}, "rented": {"eq": False},
             **({"verified": {"eq": True}} if verified_only else {}),
             "external": {"eq": False}, "reliability2": {"gte": min_reliability},
             **({"cuda_max_good": {"gte": min_cuda}} if min_cuda else {}),
             "inet_down": {"gte": min_down_mbps},
             **({"geolocation": {"notin": list(exclude_countries)}} if exclude_countries else {}),
             "direct_port_count": {"gte": 1}, "disk_space": {"gte": disk_gb}, "type": "on-demand",
             "order": [["dph_total", "asc"]], "limit": 100, "allocated_storage": disk_gb}
        # a host vast has deverified fails to start containers (seen 2026-11-17: its GPUs cannot be attached)
        return [o for o in self.call("POST", "/bundles/", q)["offers"] if o.get("verification") != "deverified"]

    def create(self, offer_id, image: str, disk_gb: int, label: str) -> str:
        r = self.call("PUT", f"/asks/{offer_id}/", {"client_id": "me", "image": image, "disk": disk_gb, "label": label,
                                                    "runtype": "ssh_direc ssh_proxy", "env": {}})
        if not r.get("success"):
            raise VastError(f"create on offer {offer_id}: {r}")
        return str(r["new_contract"])

    def show(self, iid) -> dict | None:
        return self.call("GET", f"/instances/{iid}/", query={"owner": "me"}).get("instances")

    def destroy(self, iid):
        self.call("DELETE", f"/instances/{iid}/", {})

    def ensure_ssh_key(self, public_key: str):
        """The orchestrator's key on the account: vast installs the account's keys for root on every new instance."""
        listed = self.call("GET", "/ssh/")
        rows = listed if isinstance(listed, list) else listed.get("keys") or listed.get("ssh_keys") or []
        if not any((k.get("public_key") or k.get("ssh_key") or "").split()[:2] == public_key.split()[:2] for k in rows):
            self.call("POST", "/ssh/", {"ssh_key": public_key})


class Fleet:
    def __init__(self, api: VastAPI, db, *, image: str, disk_gb: int, ssh_key: str, bundle, state_dir: Path,
                 min_reliability=0.95, min_down_mbps=200, exclude_countries=("CN",), verified_only=False,
                 min_cuda=0.0, setup_minutes=45,
                 proxy_command=None, clock=time.time, remote=None, bundle_url=None, items=None, min_credit=20.0):
        """bundle: () -> path of the setup bundle (built once). items: () -> path of the benchmark items (built once),
        given to evaluation machines only. bundle_url: (path) -> a URL the machine downloads it
        from (else it is copied over ssh, which on a slow link holds up the caller). remote: (ssh, scp) for tests."""
        self.api, self.db, self.image, self.disk_gb, self.key = api, db, image, disk_gb, os.path.expanduser(ssh_key)
        self.bundle, self.dir, self.min_rel, self.min_down = bundle, Path(state_dir), min_reliability, min_down_mbps
        self.exclude, self.verified_only, self.min_cuda = tuple(exclude_countries), verified_only, min_cuda
        self.setup_s, self.proxy, self.clock = setup_minutes * 60, proxy_command, clock
        self.ssh, self.scp = remote or (ssh, scp)
        self.bundle_url, self.items = bundle_url, items
        self._running_since = {}      # purpose -> when vast.ai first reported the container running
        self.dir.mkdir(parents=True, exist_ok=True)
        self._key_on_account = False
        self.min_credit, self._credit = min_credit, (None, 0.0)     # (credit, when read)

    def credit(self, max_age=300) -> float | None:
        """The account's credit in USD (read at most every max_age s; None when vast.ai does not answer)."""
        value, at = self._credit
        if value is None or self.clock() - at > max_age:
            try:
                value = float((self.api.call("GET", "/users/current/") or {}).get("credit"))
            except Exception:
                return value
            self._credit = (value, self.clock())
            if value < self.min_credit:
                log.warning("vast.ai credit $%.2f is under $%.0f: top it up (no machine is rented meanwhile)",
                            value, self.min_credit)
        return value

    # ---- queries
    def get(self, purpose) -> dict | None:
        return next((m for m in self.db.machines() if m["name"] == purpose), None)

    def active(self) -> list:
        return self.db.machines()

    def machine(self, row) -> Machine:
        return Machine(row["name"], row["ssh_host"], row["ssh_port"], str(self.dir / f"{row['provider_id']}.known_hosts"),
                       self.proxy)

    # ---- renting
    def rent(self, purpose: str, gpu_name: str, gpus: int):
        """The cheapest suitable offer (never a vast machine whose setup failed before). Raises LowCredit while the
        account's credit is under min_credit: its machines stop when it runs out, so none is added."""
        credit = self.credit()
        if credit is not None and credit < self.min_credit:
            raise LowCredit(f"vast.ai credit ${credit:.2f} is under ${self.min_credit:.0f}: no machine is rented "
                            "until it is topped up")
        if not self._key_on_account:
            self.api.ensure_ssh_key(Path(self.key + ".pub").read_text().strip())
            self._key_on_account = True
        bad = {m["machine_id"] for m in self.db.machines(states=("gone",)) if m["note"] and m["note"].startswith("failed")}
        offers = [o for o in self.api.offers(gpu_name, gpus, self.min_rel, self.disk_gb, self.min_down, self.exclude,
                                                  self.verified_only, self.min_cuda)
                  if str(o["machine_id"]) not in bad]
        for o in offers[:5]:                    # an offer can be taken between the search and the order
            try:
                iid = self.api.create(o["id"], self.image, self.disk_gb, f"axonry {purpose}")
                break
            except VastError as e:
                if not any(t in str(e) for t in ("no_such_ask", "not available", "already rented")):
                    raise
        else:
            raise VastError(f"no {gpus} x {gpu_name} offer on vast.ai right now")
        self.db.add_machine(purpose, iid, o["machine_id"], gpu_name, gpus, o["dph_total"], self.clock())
        log.info("rented %s: instance %s, %d x %s on machine %s at $%.3f/h", purpose, iid, gpus, gpu_name,
                 o["machine_id"], o["dph_total"])

    def advance(self, purpose) -> str:
        """Move a machine along booting -> setting_up -> ready. -> its state, or "failed" (then it is gone).
        A failed ssh/scp or API call is retried on the next call (connections drop); only setup.sh's own failure or
        the setup deadline give the machine up."""
        row = self.get(purpose)
        if row is None:
            return "gone"
        if row["state"] == "ready":
            return "ready"
        try:
            if self.clock() - row["created_at"] > self.setup_s:
                raise SetupFailed(f"not ready within {self.setup_s // 60} min ({row['state']}: {self._stuck(row)})")
            if row["state"] == "booting":
                inst = self.api.show(row["provider_id"]) or {}
                port = ((inst.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort")
                msg = (inst.get("status_msg") or "").strip()
                if inst.get("actual_status") != "running" and msg.startswith("Error"):   # e.g. its GPUs unusable
                    raise SetupFailed(f"vast.ai could not start the container: {msg[:300]}")
                if inst.get("actual_status") != "running" or not port:
                    return "booting"
                since = self._running_since.setdefault(purpose, self.clock())
                try:
                    m = self._reachable(purpose, [(inst["public_ipaddr"].strip(), int(port)),
                                                  (inst.get("ssh_host"), inst.get("ssh_port"))])
                except MachineError as e:     # running, yet no sshd answers: a broken host, not a slow one
                    if self.clock() - since > UNREACHABLE_S and "Permission denied (publickey)" not in str(e):
                        raise SetupFailed(f"running but unreachable over ssh for {UNREACHABLE_S // 60} min: "
                                          f"{str(e)[:200]}") from e
                    raise
                bundle = Path(self.bundle())
                fetch = ""
                if self.bundle_url:   # the machine downloads it in the background: a far link holds up nothing
                    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
                    fetch = (f"python3 -c {shlex.quote(FETCH)} {shlex.quote(self.bundle_url(bundle))} "
                             f"/root/axonry-setup.tar {sha} && ")
                else:
                    self.scp(m, self.key, bundle, "/root/axonry-setup.tar")
                items = ""
                if is_eval(purpose):   # the benchmark items: an evaluation machine's only (the toolkit has none)
                    self.scp(m, self.key, Path(self.items()), "/root/axonry-items.tar")
                    items = " /root/axonry-items.tar"
                setup = (f"rm -rf /root/axonry-setup {ROOT}/FAILED && {fetch}mkdir /root/axonry-setup && "
                         "tar xf /root/axonry-setup.tar -C /root/axonry-setup && "
                         f"AXONRY_GPU_CHECK=1 /root/axonry-setup/setup.sh {row['gpus']}{items} "
                         f"|| {{ mkdir -p {ROOT}; touch {ROOT}/FAILED; }}")
                self.ssh(m, self.key, f"(setsid nohup sh -c {shlex.quote(setup)} > /root/axonry-setup.log 2>&1 "
                                      "< /dev/null &)", timeout=300)
                self.db.set_machine(purpose, state="setting_up")
                return "setting_up"
            m = self.machine(row)
            st = self.ssh(m, self.key, f"cat {ROOT}/READY 2>/dev/null || (test -f {ROOT}/ITEMS_FAILED && echo ITEMS && "
                                       f"tail -c 600 /root/axonry-setup.log) || (test -f {ROOT}/FAILED && "
                                       "echo FAILED && tail -c 600 /root/axonry-setup.log) || true", timeout=45,
                          attempts=1).strip()
            if st.startswith("ITEMS"):     # our benchmark items failed their check there: not the machine's fault
                log.warning("machine %s (instance %s): benchmark items check failed: %s", purpose,
                            row["provider_id"], st[5:].strip()[-300:])
                self.release(purpose, "released: the benchmark items failed their check (ours)")
                return "failed"
            if st.startswith("FAILED"):
                raise SetupFailed(f"setup failed: {st[6:]}")
            if st:
                self.db.set_machine(purpose, state="ready", ready_at=self.clock())
                log.info("machine %s ready (instance %s)", purpose, row["provider_id"])
                return "ready"
            return "setting_up"
        except SetupFailed as e:
            log.warning("machine %s (instance %s): %s", purpose, row["provider_id"], e)
            self.release(purpose, f"failed: {e}"[:500])
            return "failed"
        except MachineError as e:              # sshd not up yet, a dropped connection, the API hiccuping
            if "Permission denied (publickey)" in str(e) and self.clock() - row["created_at"] > KEY_GRACE:
                log.warning("machine %s (instance %s): the host never installed our key", purpose, row["provider_id"])
                self.release(purpose, "failed: the host never installed the account's ssh key")
                return "failed"
            log.info("machine %s (instance %s): %s (retrying)", purpose, row["provider_id"], str(e)[:200])
            return self.get(purpose)["state"]

    def _stuck(self, row) -> str:
        """Where a machine that ran out of time stands: vast's status while booting, the setup log's tail after."""
        try:
            if row["state"] == "booting":
                inst = self.api.show(row["provider_id"]) or {}
                return f"{inst.get('actual_status')}, {(inst.get('status_msg') or '').strip()[:200]}"
            return self.ssh(self.machine(row), self.key, "tail -c 300 /root/axonry-setup.log", timeout=30,
                            attempts=1).strip().replace("\n", " | ")
        except Exception as e:     # only a diagnosis: the machine is given up either way
            return f"unknown ({str(e)[:100]})"

    def _reachable(self, purpose, endpoints) -> Machine:
        """The first endpoint whose sshd answers: the host's direct port, else vast's ssh relay (some hosts' port
        forwarding is broken). Probes try once; the next tick retries."""
        err = None
        for host, port in endpoints:
            if not host or not port:
                continue
            self.db.set_machine(purpose, ssh_host=host, ssh_port=int(port))
            m = self.machine(self.get(purpose))
            try:
                self.ssh(m, self.key, "true", timeout=45, attempts=1)
                return m
            except MachineError as e:
                if "Permission denied (publickey)" in str(e):
                    raise                     # sshd answered: the key is the problem, not the route
                err = e
        raise err or VastError("no ssh endpoint yet")

    def release(self, purpose, note="released"):
        self._running_since.pop(purpose, None)
        row = self.get(purpose)
        if row is None:
            return
        try:
            self.api.destroy(row["provider_id"])
        except VastError as e:
            if "404" not in str(e):
                raise                          # kept active: retried by the next release
        self.db.set_machine(purpose, state="gone", ended_at=self.clock(), note=note)
        Path(self.dir / f"{row['provider_id']}.known_hosts").unlink(missing_ok=True)
        log.info("machine %s released (instance %s): %s", purpose, row["provider_id"], note)

    def hand_over(self, purpose, public_key: str, deadline: int) -> Machine:
        """A ready run machine goes to this run's agent: its login key, its deadline (axonry-time). -> the machine
        as the agent reaches it."""
        m = self.machine(self.get(purpose))
        give_to_agent(m, self.key, public_key, deadline, self.ssh)
        return Machine(m.name, m.ip, m.port, m.known_hosts, None, "agent")

    def probe(self, purpose) -> dict:
        return probe(self.machine(self.get(purpose)), self.key, "open", self.ssh)

    def vanished(self, purpose, row=None) -> bool:
        """A ready machine whose instance vast no longer runs (host failure, outbid, deleted)."""
        row = row or self.get(purpose)
        if row is None or row["state"] != "ready":
            return False
        try:
            inst = self.api.show(row["provider_id"])
        except VastError:
            return False                       # the API is down, not the machine
        return inst is None or inst.get("actual_status") in ("exited", "offline", "stopped", "unknown") \
            or inst.get("intended_status") == "stopped"

