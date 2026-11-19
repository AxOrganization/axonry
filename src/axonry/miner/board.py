"""axonry-miner board: the subnet at a glance, from public data only (status.json, models/<base>/line.json).

  1. Champion and queue   the champion (none yet: the base model, weight burned at uid 0), then every
                          submission in the queue, in priority order; a run shows its time left, GPU use and the
                          agent's progress line
  2. Not in queue         submissions waiting (no PR yet, fee not covered, ...) or stopped before a run (and why),
                          PRs that carry no submission (and why), and registered miners with a funded account but no
                          commitment
  3. Recent matches       the last N decisions; per benchmark "score[not-worse at, win at]" against the champion
                          of that moment (green: win, red: worse, plain: not worse)

axonry-miner board MINER (a uid, hotkey, coldkey or submission id): that miner in full - account, entry, its run live
(time left, progress, each GPU), timeline, admission and the result per benchmark.

One registration, one entry: a hotkey's first commitment is its submission for good. A uid is shown while the
hotkey is registered. Being deregistered after committing does not stop a submission; it only means no weight: a
deregistered champion's weight is burned (uid 0).
"""
import os
import re
import sys
import time
import unicodedata

from ..common import layout

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, YELLOW, CYAN, MAGENTA = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[35m"
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ANSI = re.compile(r"\033\[[0-9;]*m|\033\]8;;[^\033]*\033\\")

STAGE = {   # status -> (label, colour)
    "REVIEW": ("in review", YELLOW), "QUEUED": ("queued", CYAN), "STARTING": ("starting", CYAN),
    "RUNNING": ("running", MAGENTA),
    "COLLECTING": ("collecting", MAGENTA), "ADMITTED": ("evaluating", MAGENTA),
    "SCORED": ("scored", MAGENTA)}
WHY = {     # reason code -> short label
    "NO_PR": "no PR yet", "UNCOMMITTED": "not committed", "PARTICIPATED": "hotkey used",
    "INSUFFICIENT_FUNDS": "insufficient funds", "COPY": "copy",
    "HOTKEY_UNKNOWN": "hotkey unknown", "BAD_PR": "bad PR", "BAD_ENTRY": "bad entry.json",
    "BASE_NOT_ALLOWED": "wrong base", "HOTKEY_MOVED": "hotkey changed owner", "REVEAL_EXPIRED": "revealed too late",
    "BAD_PARENT": "bad parent", "REVIEW_CHANGES": "review: delete and push",
    "CONFLICT": "conflict"}


def clean(text, limit=None):
    """Miner-written text (titles, reasons) never reaches the terminal with control characters."""
    t = _CTRL.sub(" ", str(text or "")).strip()
    return t if limit is None or len(t) <= limit else t[:limit - 1] + "…"


def width(s):
    s = _ANSI.sub("", s)
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


class Style:
    def __init__(self, color):
        self.color = color

    def __call__(self, text, *codes):
        codes = [c for c in codes if c]
        return "".join(codes) + text + RESET if self.color and codes else text

    def link(self, text, url):
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\" if self.color and url else text


def table(title, headers, rows, st, right=()):
    """Aligned table; cells may carry colour and links (measured without them)."""
    if not rows:
        return f"{st(title, BOLD)}\n  (none)\n"
    ws = [max(width(h), *(width(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells, head=False):
        out = []
        for i, c in enumerate(cells):
            pad = " " * (ws[i] - width(c))
            out.append(pad + c if i in right else c + pad)
        text = "  ".join(out).rstrip()
        return "  " + (st(text, BOLD) if head else text)
    rule = "  " + st("─" * (sum(ws) + 2 * (len(ws) - 1)), DIM)
    return "\n".join([st(title, BOLD), line(headers, True), rule, *(line(r) for r in rows)]) + "\n"


def board(status: dict, line: dict | None, n=20, color=None) -> str:
    st = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR") if color is None else color)
    miners = {m["hotkey"]: m for m in status.get("miners", [])}
    repo = status.get("repo")

    def uid(hk):
        m = miners.get(hk)
        return str(m["uid"]) if m else st("-", DIM)

    def hot(hk):
        return hk[:10] + "…" if hk else "-"

    def pr(n_, title):
        if not n_:
            return clean(title, 28) if title else st("-", DIM)
        url = f"https://github.com/{repo}/pull/{n_}" if repo else None
        return st.link(f"#{n_}", url) + (" " + clean(title, 28) if title else "")

    # 1. champion and queue
    rows = []
    champ = status.get("champion")
    if champ:
        online = champ["hotkey"] in miners
        detail = "receives the weight" if online else st("deregistered: weight burned (uid 0)", RED)
        rows.append([st("★", YELLOW), uid(champ["hotkey"]), hot(champ["hotkey"]),
                     champ["sid"], pr(champ.get("pr"), champ.get("title")), st("CHAMPION", BOLD, YELLOW), detail])
    else:
        rows.append([st("★", YELLOW), st("-", DIM), "-", "-", "base model", st("CHAMPION", BOLD, YELLOW),
                     "no champion yet: weight burned (uid 0)"])
    queue = sorted(status.get("running", []) + status.get("evaluating", []) + status.get("queue", [])
                   + status.get("review", []), key=lambda x: tuple(x.get("priority") or (1 << 62, 0)))
    now = status.get("updated_at") or time.time()
    for i, x in enumerate(queue, 1):
        label, col = STAGE.get(x["status"], (x["status"].lower(), ""))
        run = (miners.get(x["hotkey"]) or {}).get("run") or {}
        live = " · ".join(t for t in (f"{_left(run.get('deadline'), now)} left", clean(run.get("note"))) if t)
        detail = {"RUNNING": clean(live, 70),
                  "QUEUED": f"position {x.get('position', '-')}: waiting for a machine",
                  "STARTING": "its machine is being set up",
                  "REVIEW": "waiting for the reviewer", "COLLECTING": "run over: edit upload and admission",
                  "ADMITTED": "benchmarks on the evaluation machine",
                  "SCORED": "decided once every earlier submission is"}.get(x["status"], "")
        rows.append([str(i), uid(x["hotkey"]), hot(x["hotkey"]), x["sid"], pr(x["pr"], x.get("title")),
                     st(label, col), detail])
    out = [table("1. Champion and queue (priority order)", ["#", "UID", "HOTKEY", "SID", "PR", "STATUS", "DETAIL"],
                 rows, st, right=(0, 1))]

    # 2. not in queue
    rows, seen = [], set()
    for x, col in [(w, YELLOW) for w in status.get("waiting", [])] + [(w, RED) for w in status.get("stopped", [])]:
        seen.add(x["hotkey"])
        rows.append([uid(x["hotkey"]), hot(x["hotkey"]), x["sid"], pr(x["pr"], x.get("title")),
                     st(WHY.get(x["reason_code"], (x["reason_code"] or "").lower()), col), clean(x.get("reason"), 70)])
    for w in status.get("prs", []):
        rows.append([uid(w["hotkey"]) if w.get("hotkey") else st("-", DIM), hot(w.get("hotkey")), st("-", DIM),
                     pr(w["pr"], w.get("title")), st(WHY.get(w["reason_code"], w["reason_code"]), DIM),
                     clean(w.get("reason"), 70)])
    active = {m["hotkey"] for m in status.get("miners", []) if m.get("entry")}   # took part (one entry each)
    accounts = status.get("accounts", {})
    for m in status.get("miners", []):
        bal = (accounts.get(m.get("coldkey")) or {}).get("available_usd", 0)
        if m["hotkey"] not in seen | active and bal > 0:
            rows.append([str(m["uid"]), hot(m["hotkey"]), st("-", DIM), st("-", DIM), st("not committed", DIM),
                         f"account: {bal:.2f} USD available"])
    out.append(table("2. Not in queue", ["UID", "HOTKEY", "SID", "PR", "STATUS", "DETAIL"], rows, st, right=(0,)))

    # 3. recent matches
    decided = sorted((r for r in (line or {}).get("entries", []) if r.get("scores")),
                     key=lambda r: r.get("decided_at") or 0, reverse=True)[:n]
    benches = list(dict.fromkeys(k for r in decided for k in r["scores"]))
    rows = []
    for r in decided:
        cells = []
        for k in benches:
            acc, tie_at, win_at, verdict = r["scores"].get(k, [None] * 4)
            text = "-" if acc is None else f"{acc:.1f}[{tie_at:.1f},{win_at:.1f}]"
            cells.append(st(text, {"win": GREEN, "lose": RED}.get(verdict, "")))
        won = r.get("decision") == "NEW_CHAMPION"
        rows.append([_ts(r.get("decided_at")), uid(r["hotkey"]), hot(r["hotkey"]), r["sid"], pr(r.get("pr"), None),
                     "base" if (r.get("reference") or "").startswith("base:") else (r.get("reference") or "-"), *cells,
                     st("NEW CHAMPION", BOLD, GREEN) if won else st(f"no ({r.get('wins', 0)} wins)", DIM)])
    heads = ["DECIDED (UTC)", "UID", "HOTKEY", "SID", "PR", "VS", *(k.split("@")[0] for k in benches), "RESULT"]
    out.append(table(f"3. Recent matches (last {n}): score[not-worse at, win at] vs the champion of the moment",
                     heads, rows, st, right=(1,)))
    out.append(st(f"status {_ts(status.get('updated_at'))} UTC · a uid is shown "
                  "while the hotkey is registered · one miner in full: axonry-miner board <uid|hotkey|sid>", DIM))
    return "\n".join(out)


def _ts(t):
    return time.strftime("%m-%d %H:%M", time.gmtime(t)) if t else "-"


def _left(deadline, now):
    m = max(int((deadline or now) - now), 0) // 60
    return f"{m // 60}h{m % 60:02d}m"


# ---- one miner in full ------------------------------------------------------------------------------------------
SID = re.compile(r"^\d{10}-\d{4}$")


def find(status, line, key):
    """-> (miner row or None, sid or None, coldkey or None) for a uid, hotkey (or its first 6+ characters), coldkey or
    submission id."""
    miners = status.get("miners", [])
    rows = (line or {}).get("entries", [])
    if SID.match(key):
        hk = next((r["hotkey"] for r in rows if r["sid"] == key), None)
        m = next((m for m in miners if hk and m["hotkey"] == hk), None)
        return m, key, m and m["coldkey"]
    if key.isdigit():
        m = next((m for m in miners if m["uid"] == int(key)), None)
        return m, m and m.get("entry"), m and m["coldkey"]
    if len(key) >= 6:
        m = next((m for m in miners if m["hotkey"].startswith(key)), None)
        if m:
            return m, m.get("entry"), m["coldkey"]
        r = next((r for r in rows if (r["hotkey"] or "").startswith(key)), None)
        if r:
            return None, r["sid"], None
        if any(m["coldkey"] == key for m in miners) or key in status.get("accounts", {}):
            return None, None, key
    return None, None, None


def detail(reader, status, line, key, color=None) -> str:
    st = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR") if color is None else color)
    m, sid, coldkey = find(status, line, key)
    if not (m or sid or coldkey):
        return f"no miner, coldkey or submission matches {key!r}"
    now = status.get("updated_at") or time.time()
    out = []

    def head(text):
        out.append("\n" + st(text, BOLD))

    def kv(k, v):
        out.append(f"  {k:<11} {v}")

    head("MINER")
    if m:
        kv("uid", str(m["uid"]))
        kv("hotkey", m["hotkey"])
    kv("coldkey", coldkey or "-")
    others = [x for x in status.get("miners", []) if coldkey and x["coldkey"] == coldkey and x is not m]
    if others:
        kv("hotkeys" if not (m or sid) else "also", ", ".join(
            f"uid {x['uid']} " + (x.get("entry") or "no entry") for x in others))
    a = status.get("accounts", {}).get(coldkey) if coldkey else None
    head("ACCOUNT (USD)")
    if a:
        kv("available", st(f"{a['available_usd']:.2f}", BOLD) + f"   deposited {a['deposited_usd']:.2f} · charged "
           f"{a['charged_usd']:.2f} · held for runs {a['held_usd']:.2f}")
        for asset, v in sorted((a.get("pending_rao") or {}).items()):
            kv("pending", f"{v / 1e9:.6f} {asset} (not priced yet)")
    else:
        kv("", st("no deposits seen yet", DIM))
    if not sid:
        if m:
            head("ENTRY")
            kv("", st("none: this hotkey has not committed", DIM))
        return "\n".join(out).lstrip("\n")
    sub = reader.get(layout.submission(sid, "submission.json")) or {}
    live = next((r for k in ("review", "queue", "running", "evaluating", "waiting") for r in status.get(k, [])
                 if r["sid"] == sid), None)
    if live and not sub.get("status"):          # an entry still in progress: its live row
        sub = {**sub, "status": live["status"], "reason_code": live.get("reason_code"), "reason": live.get("reason"),
               "pr": sub.get("pr") or ({"number": live["pr"], "title": live.get("title")} if live.get("pr") else None)}
    status_ = sub.get("status") or next((r["status"] for r in (line or {}).get("entries", []) if r["sid"] == sid), "?")
    head(f"ENTRY {sid}")
    kv("status", st(f"{status_} {sub.get('reason_code') or ''}".strip(), BOLD))
    if sub.get("reason"):
        kv("reason", clean(sub["reason"], 300))
    pr = sub.get("pr") or {}
    facts = [f"PR #{pr['number']} {clean(pr.get('title'), 50)}" if pr.get("number") else None,
             f"commit {pr['locked_commit'][:12]}" if pr.get("locked_commit") else None,
             f"base {sub['base']}" if sub.get("base") else None, f"parent {sub['parent']}" if sub.get("parent") else None]
    if any(facts):
        kv("pr", " · ".join(f for f in facts if f))
    fee = sub.get("fee") or {}
    if fee.get("gpus"):                         # locked: the run's terms
        agent = f" + agent up to {fee['max_agent_usd']} USD" if fee.get("max_agent_usd") else ""
        kv("fee", f"{fee['gpus']} GPU x {fee['usd_per_gpu_hour']} USD/GPU-hour{agent} · held {fee['max_usd']:.2f} · "
                  f"charged {fee.get('charged_usd', 0):.2f}")
    elif fee:                                   # refused before its run: what it was charged (the review fee)
        kv("fee", f"charged {fee.get('charged_usd', 0):.2f}")
    run = (m or {}).get("run")
    if run and run.get("sid") == sid:
        head(f"RUN (live, sampled {_ts(run.get('sampled_at'))} UTC)")
        kv("deadline", f"{_ts(run.get('deadline'))} UTC · {_left(run.get('deadline'), now)} left")
        kv("progress", clean(run.get("note")) or st("no progress line yet", DIM))
        for i, g in enumerate(run.get("gpus") or []):
            u = g["util_pct"]
            bar = st("█" * (u // 5), GREEN if u >= 50 else YELLOW) + st("░" * (20 - u // 5), DIM)
            kv("GPU " + str(i), f"{bar} {u:>3}%   {g['mem_used_mib'] / 1024:5.1f} / {g['mem_total_mib'] / 1024:.1f} GiB")
    tl = sub.get("timeline") or {}
    events = sorted((v, k) for k, v in tl.items() if v and (k != "deadline" or "agent_end" not in tl))
    if events:
        head("TIMELINE (UTC)")
        for t, k in events:
            kv(_ts(t), k.replace("_", " "))
    adm = reader.get(layout.submission(sid, "admission.json"))
    if adm:
        head("ADMISSION " + (st("passed", GREEN) if adm["passed"] else st("rejected", RED)))
        for v in adm.get("violations") or []:
            kv("-", clean(v, 200))
    score = reader.get(layout.submission(sid, "score.json"))
    if score:
        bs = score["benchmarks"]
        ref = next(iter(bs.values()))["vs_champion"]["ref"]
        won = score["decision"] == "NEW_CHAMPION"
        head(f"RESULT {st(score['decision'], GREEN if won else '')} vs {ref}: {score['wins']} wins "
             f"(needs {score['judge']['n_win']}), not worse on all: {'yes' if score['tie_all'] else 'no'}")
        rows = []
        for k, b in bs.items():
            c = b["vs_champion"]
            rows.append([k.split("@")[0], f"{b['acc']:.2f}", f"{c['ref_acc']:.2f}", f"{c['d']:+.2f}", f"{c['se']:.2f}",
                         f"{c['tie_at']:.2f}", f"{c['win_at']:.2f}",
                         st("win", GREEN) if c["win"] else (st("not worse", "") if c["tie_ok"] else st("worse", RED))])
        out.append(table("", ["BENCHMARK", "ACC", "CHAMPION", "D", "SE", "NOT-WORSE AT", "WIN AT", "VERDICT"], rows,
                         st, right=(1, 2, 3, 4, 5, 6)).split("\n", 1)[1].rstrip())
    ruling = reader.get(layout.submission(sid, "ruling.json"))
    if ruling:
        head(st("DISQUALIFIED by the owner", RED))
        kv("", clean(ruling["reason"], 300))
    if sub.get("agent_log") == "published" or adm or score:
        head("FILES")
        kv("public", f"{reader.base or '<public>'}/{layout.submission(sid, '')}"
               + ("  (agent/: the agent's full log)" if sub.get("agent_log") == "published" else ""))
    return "\n".join(out).lstrip("\n")


DASHBOARD = "https://axonry.ai/miner-dashboard/"


def cmd_board(reader, n=20, color=None, watch=0, miner=None):
    """watch: redraw every that many seconds (the subnet publishes its status about once a minute); miner: one miner
    in full instead of the overview."""
    while True:
        status = reader.get(layout.STATUS)
        line = reader.get(layout.line(status["base"])) if status and status.get("base") else None
        if not status:
            text = "no status published yet"
        elif miner:
            text = detail(reader, status, line, miner, color)
        else:
            text = board(status, line, n, color)
        if status and reader.base:
            text += (f"\n\nMore detail: {DASHBOARD}\nData: {reader.base}/{layout.STATUS}")
        print(("\033[H\033[2J" if watch else "") + text, flush=True)
        if not watch:
            return
        time.sleep(watch)
