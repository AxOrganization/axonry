"""Orchestrator state (SQLite; one writer process). The public bucket is derived from it; the private bucket keeps
periodic snapshots (state/)."""
import json
import sqlite3
import tempfile
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  sid TEXT PRIMARY KEY, base TEXT, hotkey TEXT, coldkey TEXT, status TEXT NOT NULL, reason_code TEXT, reason TEXT,
  content_digest TEXT, record TEXT NOT NULL, created_at REAL, updated_at REAL, decided_at REAL,
  infra_retries INTEGER DEFAULT 0,
  pr_number INTEGER, head_sha TEXT, prio_block INTEGER, prio_index INTEGER,
  locked_commit TEXT, run_cost INTEGER);
DROP INDEX IF EXISTS one_per_hotkey;
CREATE INDEX IF NOT EXISTS by_hotkey ON submissions(hotkey);
CREATE INDEX IF NOT EXISTS by_digest ON submissions(content_digest);
CREATE TABLE IF NOT EXISTS deposits (block INTEGER, idx INTEGER, coldkey TEXT, asset TEXT, amount INTEGER, ts INTEGER,
  usd INTEGER, price TEXT, PRIMARY KEY (block, idx, asset));
CREATE TABLE IF NOT EXISTS charges (sid TEXT, n INTEGER, coldkey TEXT, usd INTEGER, at REAL, PRIMARY KEY (sid, n));
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS machines (provider_id TEXT PRIMARY KEY, name TEXT, machine_id TEXT, gpu_name TEXT,
  gpus INTEGER, usd_hour REAL, state TEXT, ssh_host TEXT, ssh_port INTEGER, created_at REAL, ready_at REAL,
  ended_at REAL, note TEXT);
CREATE TABLE IF NOT EXISTS runs (sid TEXT PRIMARY KEY, task TEXT, handle TEXT, sup_state TEXT);
CREATE TABLE IF NOT EXISTS results (model_id TEXT, bench_key TEXT, scores TEXT, PRIMARY KEY (model_id, bench_key));
CREATE TABLE IF NOT EXISTS champions (base TEXT, seq INTEGER, sid TEXT, hotkey TEXT, block INTEGER, at REAL,
  evidence TEXT, PRIMARY KEY (base, seq));
"""

class DB:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(str(path), isolation_level=None, timeout=60)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")      # main loop and eval worker are separate processes
        self.c.executescript(SCHEMA)

    def snapshot(self) -> bytes:
        """A consistent copy of the whole database (the WAL included), taken while it is in use."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "snapshot.sqlite"
            self.c.execute("VACUUM INTO ?", (str(out),))
            return out.read_bytes()

    # ---- kv
    def get(self, k, default=None):
        r = self.c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def set(self, k, v):
        self.c.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    # ---- submissions
    def add_submission(self, sid, base, hotkey, coldkey, status, record, content_digest, prio):
        """A hotkey's submission (one registration, one entry; the owner's hotkey may enter more than once)."""
        now = time.time()
        self.c.execute("INSERT OR IGNORE INTO submissions(sid,base,hotkey,coldkey,status,content_digest,record,"
                       "created_at,updated_at,prio_block,prio_index) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (sid, base, hotkey, coldkey, status, content_digest, json.dumps(record), now, now, *prio))

    def of_hotkey(self, hotkey):
        r = self.c.execute("SELECT * FROM submissions WHERE hotkey=? ORDER BY prio_block DESC, prio_index DESC",
                           (hotkey,)).fetchone()
        return self._row(r) if r else None

    def first_with_digest(self, digest):
        """The earliest submission (chain order) committing this content, or None."""
        r = self.c.execute("SELECT * FROM submissions WHERE content_digest=? ORDER BY prio_block, prio_index LIMIT 1",
                           (digest,)).fetchone()
        return self._row(r) if r else None

    def sub(self, sid):
        r = self.c.execute("SELECT * FROM submissions WHERE sid=?", (sid,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r):
        d = dict(r)
        d["record"] = json.loads(d["record"])
        return d

    def subs(self, status=None, base=None):
        q, args = "SELECT * FROM submissions WHERE 1=1", []
        if status:
            q += f" AND status IN ({','.join('?' * len(status))})"
            args += list(status)
        if base:
            q += " AND base=?"
            args.append(base)
        return [self._row(r) for r in self.c.execute(q + " ORDER BY sid", args)]

    def update(self, sid, status=None, record_patch=None, **cols):
        s = self.sub(sid)
        rec = {**s["record"], **(record_patch or {})}
        sets = {"record": json.dumps(rec), "updated_at": time.time(), **cols}
        if status:
            sets["status"] = status
        self.c.execute(f"UPDATE submissions SET {','.join(f'{k}=?' for k in sets)} WHERE sid=?",
                       [*sets.values(), sid])

    # ---- accounts (virtual; the funds sit in the treasury), in micro-USD: a deposit counts at its own moment's price
    HOLDING = ("QUEUED", "RUNNING")     # a locked entry holds its maximum run cost, less what it has been charged

    def add_deposit(self, d):
        self.c.execute("INSERT OR IGNORE INTO deposits(block,idx,coldkey,asset,amount,ts) VALUES(?,?,?,?,?,?)",
                       (d.block, d.extrinsic_index, d.coldkey, d.asset, d.amount_rao, d.timestamp))

    def unpriced_deposits(self):
        return [dict(r) for r in self.c.execute("SELECT * FROM deposits WHERE usd IS NULL ORDER BY block, idx")]

    def price_deposit(self, block, idx, asset, usd: int, price: dict):
        self.c.execute("UPDATE deposits SET usd=?, price=? WHERE block=? AND idx=? AND asset=? AND usd IS NULL",
                       (usd, json.dumps(price), block, idx, asset))

    def charge(self, sid, n, coldkey, usd: int):
        """Charge n of an entry, once each: n = 1, 2, ... for each full GPU hour; 0 settles the GPU time at its end; -1
        is its agent's model use; -2 the review fee (when it went to review); -3 gives the review fee back when its run
        ends, as the run's own charges then cover it."""
        self.c.execute("INSERT OR IGNORE INTO charges VALUES(?,?,?,?,?)", (sid, n, coldkey, usd, time.time()))

    def set_aside(self, sid, key):
        """A stopped run's charges (GPU hours, settlement, agent use) kept on the account under `key`: the entry's own
        run is charged anew (its review fee stays with it)."""
        self.c.execute("UPDATE charges SET sid=? WHERE sid=? AND n >= -1", (key, sid))

    def refund(self, sid):
        """Our infrastructure lost the run: everything it was charged goes back (it runs again from the start)."""
        self.c.execute("DELETE FROM charges WHERE sid=?", (sid,))

    def charged(self, sid) -> dict:
        """{n: micro-USD} of one run."""
        return {r["n"]: r["usd"] for r in self.c.execute("SELECT n, usd FROM charges WHERE sid=?", (sid,))}

    def available(self, coldkey, besides=None) -> int:
        """Balance minus what the coldkey's other locked entries still hold (micro-USD)."""
        a = self._account(coldkey, besides)
        return a["balance"] - a["held"]

    def _account(self, coldkey, besides=None):
        one = lambda q, *args: self.c.execute(q, args).fetchone()[0] or 0  # noqa: E731
        dep = one("SELECT SUM(usd) FROM deposits WHERE coldkey=? AND usd IS NOT NULL", coldkey)
        chg = one("SELECT SUM(usd) FROM charges WHERE coldkey=?", coldkey)
        held = one(f"SELECT SUM(s.run_cost - COALESCE((SELECT SUM(usd) FROM charges c WHERE c.sid=s.sid), 0)) "
                   f"FROM submissions s WHERE s.coldkey=? AND s.status IN {self.HOLDING} AND s.sid IS NOT ?",
                   coldkey, besides)
        pending = {r["asset"]: r["n"] for r in self.c.execute(
            "SELECT asset, SUM(amount) AS n FROM deposits WHERE coldkey=? AND usd IS NULL GROUP BY asset", (coldkey,))}
        return {"deposited": dep, "charged": chg, "balance": dep - chg, "held": max(held, 0), "pending": pending}

    def accounts(self) -> dict:
        """{coldkey: {deposited_usd, charged_usd, balance_usd, held_usd, available_usd, pending_rao: {asset: rao}}}"""
        out = {}
        for (ck,) in self.c.execute("SELECT DISTINCT coldkey FROM deposits UNION SELECT DISTINCT coldkey FROM charges"):
            a = self._account(ck)
            usd = lambda v: round(v / 1e6, 6)  # noqa: E731
            out[ck] = {"deposited_usd": usd(a["deposited"]), "charged_usd": usd(a["charged"]),
                       "balance_usd": usd(a["balance"]), "held_usd": usd(a["held"]),
                       "available_usd": usd(a["balance"] - a["held"]), "pending_rao": a["pending"]}
        return out

    def lock(self, sid, commit, coldkey, run_cost: int, record_patch):
        """An approved submission is locked at its commit; from now on it holds its maximum run cost."""
        self.update(sid, record_patch=record_patch, locked_commit=commit, coldkey=coldkey, run_cost=run_cost)

    # ---- GPU machines (pool/: owned hosts and vast.ai)
    def add_machine(self, name, provider_id, machine_id, gpu_name, gpus, usd_hour, at):
        """name: what the machine is for (a submission id, or "eval"); at most one active machine per name."""
        self.c.execute("INSERT INTO machines(name,provider_id,machine_id,gpu_name,gpus,usd_hour,state,created_at) "
                       "VALUES(?,?,?,?,?,?,?,?)", (name, provider_id, str(machine_id), gpu_name, gpus, usd_hour,
                                                    "booting", at))

    def machines(self, states=("booting", "setting_up", "ready")):
        q = ",".join("?" * len(states))
        return [dict(r) for r in self.c.execute(f"SELECT * FROM machines WHERE state IN ({q}) ORDER BY created_at",
                                                states)]

    def set_machine(self, name, **cols):
        self.c.execute(f"UPDATE machines SET {','.join(f'{k}=?' for k in cols)} WHERE name=? AND state!='gone'",
                       (*cols.values(), name))

    # ---- runs
    def save_run(self, sid, task=None, handle=None, sup_state=None):
        cur = self.run(sid) or {}
        vals = {"task": task if task is not None else cur.get("task"),
                "handle": handle if handle is not None else cur.get("handle"),
                "sup_state": sup_state if sup_state is not None else cur.get("sup_state", {})}
        self.c.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?)",
                       (sid, json.dumps(vals["task"]), json.dumps(vals["handle"]), json.dumps(vals["sup_state"])))

    def run(self, sid):
        r = self.c.execute("SELECT * FROM runs WHERE sid=?", (sid,)).fetchone()
        return {k: json.loads(r[k]) for k in ("task", "handle", "sup_state")} if r else None

    # ---- results
    def put_results(self, model_id, bench_key, scores: dict):
        self.c.execute("INSERT OR REPLACE INTO results VALUES(?,?,?)", (model_id, bench_key, json.dumps(scores)))

    def results(self, model_id) -> dict:
        return {r["bench_key"]: json.loads(r["scores"])
                for r in self.c.execute("SELECT * FROM results WHERE model_id=?", (model_id,))}

    # ---- champions
    def champion(self, base):
        """The latest champion of the base's line that has not been disqualified (None: the base model itself)."""
        r = self.c.execute("SELECT c.* FROM champions c LEFT JOIN submissions s ON s.sid = c.sid WHERE c.base=? "
                           "AND COALESCE(s.status, '') != 'DISQUALIFIED' ORDER BY c.seq DESC LIMIT 1",
                           (base,)).fetchone()
        return dict(r) if r else None

    def add_champion(self, base, sid, hotkey, block, evidence):
        seq = self.c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM champions WHERE base=?", (base,)).fetchone()[0]
        self.c.execute("INSERT INTO champions VALUES(?,?,?,?,?,?,?)",
                       (base, seq, sid, hotkey, block, time.time(), json.dumps(evidence)))

    def champions(self, base=None):
        q = "SELECT * FROM champions" + (" WHERE base=?" if base else "") + " ORDER BY base, seq"
        return [dict(r) for r in self.c.execute(q, (base,) if base else ())]
