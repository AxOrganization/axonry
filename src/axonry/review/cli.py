"""axonry-review: the reviewer (review/PROTOCOL.md). Token: AXONRY_REVIEW_TOKEN, an account listed in the rules'
review.reviewers; nothing else.

  axonry-review run [--workdir D] [--command CMD]  the automatic reviewer: the queue top-down, POLICY.md answered by
                                                   a model (Codex by default), verdicts posted as GitHub reviews
  axonry-review queue                      the review queue in priority order (from the public status.json)
  axonry-review show PR                    the item's content at its listed commit and its digest check
  axonry-review approve PR [--body TEXT]   a human verdict (items the automatic reviewer left), bound to that commit
  axonry-review changes PR --body TEXT     request changes (the reason is public; the submission parks until an update)
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .. import registry
from ..common import layout
from ..common.digest import content_digest_of_blobs
from ..intake import binding
from ..intake.github import GitHub, check_files
from ..miner.cli import PublicReader, public_url

COMPETITION = Path(__file__).resolve().parents[3] / "WeightSpaceProgramming"


def load_queue(reader):
    st = reader.get(layout.STATUS)
    if not st:
        raise SystemExit("no status.json published yet")
    return st["review"]


def item_for(queue, pr):
    for i, x in enumerate(queue):
        if x["pr"] == pr:
            return i, x
    raise SystemExit(f"PR #{pr} is not in the review queue")


def cmd_queue(queue):
    if not queue:
        print("review queue empty")
    for i, x in enumerate(queue, 1):
        b, j = x["priority"]
        print(f"{i:>3}. PR #{x['pr']}  sid {x['sid']}  priority {b}-{j}  commit {x['commit'][:12]}  "
              f"hotkey {x['hotkey'][:12]}…")


def cmd_show(gh, rules, comp_rules, x):
    blobs = gh.dir_blobs(x["commit"], x["directory"])
    digest = content_digest_of_blobs(blobs)
    print(f"PR #{x['pr']}  commit {x['commit']}  directory {x['directory']}")
    print(f"content digest {digest}  {'== committed' if digest == x['content_digest'] else '!= COMMITTED - do not review'}")
    problem = check_files(blobs, comp_rules)
    entry = binding.check_entry(blobs.get("entry.json"), sorted({r["base"] for r in registry.all_rules()}),
                                comp_rules)
    print(f"static checks: {problem or (entry.detail if isinstance(entry, binding.Rejected) else 'pass')}")
    for rel in sorted(blobs):
        print(f"  {len(blobs[rel]):>8}  {rel}")
    for name in ("entry.json", "METHOD.md"):
        if name in blobs:
            print(f"\n----- {name} -----\n{blobs[name].decode('utf-8', 'replace')}")


def cmd_verdict(gh, queue, pr, event, body):
    _, x = item_for(queue, pr)
    if content_digest_of_blobs(gh.dir_blobs(x["commit"], x["directory"])) != x["content_digest"]:
        raise SystemExit("the listed commit no longer matches the committed content; wait for the next status")
    gh.post_review(pr, x["commit"], event, body)
    print(f"posted {event} on PR #{pr} at {x['commit'][:12]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="axonry-review", description=__doc__.split("\n")[0])
    ap.add_argument("--network", default="finney")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--workdir", type=Path, default=Path(".state/review"))
    rn.add_argument("--command", default=None, help="reviewer model command; {schema} and {out} are filled in")
    rn.add_argument("--interval", type=int, default=60)
    rn.add_argument("--once", action="store_true")
    sub.add_parser("queue")
    sub.add_parser("show").add_argument("pr", type=int)
    ap_ok = sub.add_parser("approve")
    ap_ok.add_argument("pr", type=int)
    ap_ok.add_argument("--body", default="Approved: the method may be run as committed.")
    ch = sub.add_parser("changes")
    ch.add_argument("pr", type=int)
    ch.add_argument("--body", required=True)
    a = ap.parse_args(argv)
    from ..orchestrator.cli import load_dotenv  # noqa: PLC0415
    load_dotenv()                             # AXONRY_REVIEW_TOKEN may live in .env, as the orchestrator's secrets do
    reader = PublicReader(public_url(a.network))
    rules = registry.latest_rules()
    gh = GitHub(rules["competition"]["repo"], os.environ.get("AXONRY_REVIEW_TOKEN"))
    if a.cmd == "run":
        import logging  # noqa: PLC0415

        from . import auto  # noqa: PLC0415
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        auto.run(reader, gh, rules, a.workdir, a.command or auto.DEFAULT_COMMAND, a.interval, a.once)
        return 0
    queue = load_queue(reader)
    if a.cmd == "queue":
        return cmd_queue(queue)
    comp_rules = json.loads((COMPETITION / "rules.json").read_text())
    if a.cmd == "show":
        cmd_show(gh, rules, comp_rules, item_for(queue, a.pr)[1])
    elif a.cmd == "approve":
        cmd_verdict(gh, queue, a.pr, "APPROVE", a.body)
    else:
        cmd_verdict(gh, queue, a.pr, "REQUEST_CHANGES", a.body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
