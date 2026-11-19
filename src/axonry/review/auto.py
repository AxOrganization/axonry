"""The automatic reviewer (review/PROTOCOL.md): a separate process that takes the review queue top-down, has a model
(Codex by default) answer POLICY.md on each item and posts the verdict as a GitHub review bound to the commit.

It reads only public data (status.json) and GitHub, and writes only reviews. It never touches the orchestrator's
state: the orchestrator sees the review, locks the approved commit and queues it.
"""
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .. import registry
from ..common import layout
from ..common.digest import content_digest_of_blobs

log = logging.getLogger("axonry.review")
HERE = Path(__file__).parent
POLICY, SCHEMA = HERE / "POLICY.md", HERE / "verdict.schema.json"
DEFAULT_COMMAND = ("codex exec --sandbox read-only --skip-git-repo-check --output-schema {schema} "
                   "--output-last-message {out} -")
CHECKS = ("substantive", "conduct", "original")
MAX_ATTEMPTS = 3


def earlier_than(item, contents):
    """The content the item must differ from: every earlier commitment of another hotkey (any status)."""
    p = tuple(item["priority"])
    return [c for c in contents if tuple(c["priority"]) < p and c["hotkey"] != item["hotkey"]]


def prepare(gh, item, contents, rules, work: Path) -> Path:
    """workspace/ submission/ (the item at its listed commit) + earlier/ (+ INDEX.md) + original/ (a reduced item: the
    content as first revealed, to check the deletions against); -> its path."""
    ws = (work / f"pr{item['pr']}-{item['commit'][:12]}").resolve()   # absolute: the model runs inside it
    blobs = gh.dir_blobs(item["commit"], item["directory"])
    if content_digest_of_blobs(blobs) != item["content_digest"]:
        raise ValueError("the listed commit does not hold the committed content (status.json is stale)")
    write_dir(ws / "submission", blobs)
    if item.get("revealed"):
        write_dir(ws / "original", gh.dir_blobs(item["revealed"]["commit"], item["revealed"]["directory"]))
    index = ["# Earlier submissions of other miners (earliest first)", ""]
    for c in earlier_than(item, contents):
        name = f"{c['priority'][0]}-{c['priority'][1]}-pr{c['pr']}"
        cache = work / "cache" / f"{c['commit']}-{c['directory'].strip('/').replace('/', '_')}"
        if not cache.exists():
            write_dir(cache, gh.dir_blobs(c["commit"], c["directory"]))
        dst = ws / "earlier" / name
        if not dst.exists():        # a copy, not a link: the reviewer reads nothing outside its workspace
            shutil.copytree(cache, dst)
        index.append(f"- `{name}/` hotkey {c['hotkey']} status {c['status']}")
    (ws / "earlier").mkdir(parents=True, exist_ok=True)
    (ws / "earlier" / "INDEX.md").write_text("\n".join(index) + "\n")
    benches = ", ".join(registry.load_benchmark(k).doc["id"] for k in rules.benchmark_keys())
    (ws / "PROMPT.md").write_text(POLICY.read_text() + f"\nThe evaluation benchmarks: {benches}.\n\n"
                                  "Answer with the JSON object the output schema describes. Set `read` to true when you "
                                  "read every file of the workspace (another way after a tool failed counts); false only "
                                  "if a file stayed unread.\n")
    return ws


def write_dir(root: Path, blobs: dict):
    for rel, data in blobs.items():
        p = (root / rel).resolve()
        if root.resolve() not in p.parents:
            raise ValueError(f"path escapes the workspace: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def ask_model(ws: Path, command: str, timeout=1800) -> dict:
    """Run the reviewer model in the workspace (read-only sandbox, no network for its commands, no tokens in its
    environment) and return its verdict, validated."""
    out = ws / "verdict.json"
    out.unlink(missing_ok=True)
    argv = shlex.split(command.format(schema=SCHEMA, out=out))
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k and "SECRET" not in k}
    p = subprocess.run(argv, cwd=ws, input=(ws / "PROMPT.md").read_text(), text=True, env=env, timeout=timeout,
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{argv[0]} exited {p.returncode}: {(p.stderr or p.stdout).strip()[-400:]}")
    v = json.loads(out.read_text())
    if v.get("read") is not True:         # a tool failed: no verdict on the method, never a public refusal
        raise RuntimeError(f"the reviewer could not read the workspace: {str(v.get('message'))[:300]}")
    for k in CHECKS:
        if not isinstance(v.get(k), dict) or not isinstance(v[k].get("ok"), bool):
            raise ValueError(f"verdict without {k}.ok")
    if not isinstance(v.get("message"), str) or not v["message"].strip():
        raise ValueError("verdict without a message")
    return v


FIX_MARK = "axonry-fix: "      # in a review body: whether deletions may fix a refused content (orchestrator reads it)


def fixable(body: str | None) -> bool:
    """A review asking for changes allows a fix by deletion unless it says it does not (a reviewer's own words, or an
    older review, leave it open)."""
    return f"{FIX_MARK}no" not in (body or "")


def review_body(v: dict, item: dict) -> tuple[str, str]:
    """-> (GitHub event, public body). Approved only when all three checks pass."""
    ok = all(v[k]["ok"] for k in CHECKS)
    cell = lambda t: " ".join(str(t).split()).replace("|", "\\|")       # noqa: E731 - one table cell
    lines = [f"### axonry review: {'approved' if ok else 'changes requested'}", "", v["message"].strip(), "",
             "| Check | Result | Reason |", "|---|---|---|"]
    for k in CHECKS:
        why = v[k].get("reason", "")
        if k == "original" and not v[k]["ok"] and v[k].get("copies"):
            why += f" (copies `{v[k]['copies']}`)"
        lines.append(f"| {k.capitalize()} | {'Pass' if v[k]['ok'] else '**Fail**'} | {cell(why)} |")
    fix = ok or (v.get("fixable") is True and v["original"]["ok"])      # a copy is never fixed by deleting
    if not ok:
        lines += ["", "**Fix by deletion: possible.** Delete what is named above (files, words or punctuation; keep the "
                      "idea) and push to this pull request, within the time the entry's status gives; the rest is "
                      "reviewed again." if fix else
                      "**Fix by deletion: not possible.** The entry ends here."]
    lines += ["", f"<sub>Content `{item['content_digest'][:16]}` at commit `{item['commit'][:12]}`. "
                  "Checks: substantive, conduct and original, as in the review policy of axonry.</sub>",
              f"<!-- {FIX_MARK}{'yes' if fix else 'no'} -->"]
    return ("APPROVE" if ok else "REQUEST_CHANGES"), "\n".join(lines)[:60000]


def run(reader, gh, rules, work: Path, command=DEFAULT_COMMAND, interval=60, once=False):
    """Loop: the first item of the review queue (priority order) without a verdict on its content is reviewed.
    An item whose review fails MAX_ATTEMPTS times is left to a human (axonry-review approve/changes) and skipped."""
    work.mkdir(parents=True, exist_ok=True)
    attempts_file = work / "attempts.json"
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else {}
    reviewers = rules["review"]["reviewers"]
    while True:
        try:
            st = reader.get(layout.STATUS) or {}
        except Exception as e:  # noqa: BLE001 - the bucket is unreachable for a moment: read again next round
            log.warning("status.json unreadable: %s (retrying)", str(e)[:200])
            st = {}
        for item in st.get("review", []):
            key = f"{item['pr']}:{item['content_digest']}"
            if attempts.get(key, 0) >= MAX_ATTEMPTS:
                continue
            if gh.review_state(item["pr"], item["content_digest"], item["directory"], reviewers)[0] is not None:
                continue                  # already reviewed; status.json just has not caught up
            try:
                ws = prepare(gh, item, st.get("contents", []), rules, work)
                event, body = review_body(ask_model(ws, command), item)
                gh.post_review(item["pr"], item["commit"], event, body)
                log.info("PR #%s %s at %s", item["pr"], event, item["commit"][:12])
            except Exception as e:  # noqa: BLE001 - retried; after MAX_ATTEMPTS a human decides
                attempts[key] = attempts.get(key, 0) + 1
                attempts_file.write_text(json.dumps(attempts))
                log.warning("PR #%s review attempt %d failed: %s%s", item["pr"], attempts[key], e,
                            " - left to a human" if attempts[key] >= MAX_ATTEMPTS else "")
            break                         # one item per round, then re-read the queue
        if once:
            return
        time.sleep(interval)
