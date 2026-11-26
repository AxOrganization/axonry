"""Who a submission is and what it may carry. Pure decision logic.

A miner is a hotkey: it holds the registration, signs its commitment and gets the weight. One registration is one
entry: the hotkey's first commitment on the subnet ("<content digest>:<GitHub id>", set_commitment, which the chain
takes only from a registered hotkey) is its one submission, for good. Its money is the account of the coldkey that
owned the hotkey at the commitment (fixed then; a hotkey whose owner changed since its registration cannot
enter): deposits are transfers from that coldkey.
A PR carries the submission when it is open in submissions/<hotkey>/<entry>/, by the committed GitHub id, with
exactly the committed content, and was opened after the commitment's block. A content committed earlier by another hotkey is a copy.
The content may only shrink: to fix what a review or the content checks found, a PR whose content is the current one with
files, words or punctuation deleted carries it from then on (and no earlier version does); nothing can be added or
rewritten, not even within a word.
When the rules set reveal_hours, the first PR carrying it must also be opened within that many hours of the
commitment: a commitment whose content nobody can see must not hold its priority indefinitely.
"""
import json
import re

# public, stable reason codes
NO_PR = "NO_PR"                          # no open PR carries the committed content yet
UNCOMMITTED = "UNCOMMITTED"              # a PR whose hotkey has no commitment, or does not match it
PARTICIPATED = "PARTICIPATED"            # a PR of a hotkey whose one entry is already locked or over
COPY = "COPY"                            # the committed content was committed earlier by another hotkey (final)
REVEAL_EXPIRED = "REVEAL_EXPIRED"        # no PR carried the committed content within the rules' reveal_hours (final)
HOTKEY_UNKNOWN = "HOTKEY_UNKNOWN"        # the hotkey had no owning coldkey at its commitment: nobody to charge (final)
HOTKEY_MOVED = "HOTKEY_MOVED"            # the hotkey's owner changed since its registration (final)
INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
BAD_PR = "BAD_PR"
BAD_ENTRY = "BAD_ENTRY"
BASE_NOT_ALLOWED = "BASE_NOT_ALLOWED"
CHAMPION = "champion"                    # entry.json parent: the champion when the run starts


class Rejected:
    def __init__(self, code, detail):
        self.code, self.detail = code, detail


def reduces(new: dict, old: dict) -> bool:
    """New content is the old one with deletions only: every file of new is a file of old whose words and punctuation
    marks are a subsequence of its own, in order (whitespace aside). Whole words go or stay: no word is cut into
    another; whether what is left is still the same idea is the review's call."""
    def tokens(data: bytes) -> list:
        return re.findall(rb"\w+|[^\w\s]", data)

    def subsequence(a: bytes, b: bytes) -> bool:
        rest = iter(tokens(b))
        return all(any(x == y for y in rest) for x in tokens(a))
    return bool(new) and new != old and all(k in old and subsequence(v, old[k]) for k, v in new.items())


def carrier_problem(s, p, current=None, reduced=lambda p: False) -> str | None:
    """Why PR candidate p cannot carry submission s (None: it can). current: the digest of the content s carries now
    (default: the committed one); reduced(p): p's content is an accepted reduction of it (see reduces)."""
    if p.problem or p.hotkey != s["hotkey"]:
        return p.problem or "another hotkey's directory"
    if p.author_id != s["record"]["github_id"]:
        return f"the PR is by {p.author} (GitHub id {p.author_id}); the commitment names GitHub id {s['record']['github_id']}"
    current = current or s["content_digest"]
    if p.digest != current and not reduced(p):
        return (f"the PR content {(p.digest or '')[:12]} is neither the entry's content {current[:12]} nor, while a "
                "fix is due, that content with only deletions")
    if p.opened_at is not None and p.opened_at < s["record"]["timeline"]["committed"]:
        return "the PR was opened before the commitment's block: commit first, then open the PR"
    return None


def check_entry(entry_bytes: bytes | None, bases, comp_rules: dict) -> Rejected | dict:
    """entry.json against the competition protocol (schema id: submission.entry_schema when the protocol names one)
    and the bases whose lines it may enter."""
    if entry_bytes is None:
        return Rejected(BAD_ENTRY, "entry.json missing")
    try:
        entry = json.loads(entry_bytes)
    except ValueError as e:
        return Rejected(BAD_ENTRY, f"entry.json: {e}")
    want = comp_rules["submission"].get("entry_schema")
    if not isinstance(entry, dict) or not isinstance(entry.get("base"), str) or (want and entry.get("schema") != want):
        return Rejected(BAD_ENTRY, f"entry.json needs base{f' and schema {want}' if want else ''}")
    if entry["base"] not in bases:
        return Rejected(BASE_NOT_ALLOWED, f"base {entry['base']} is not the base of a line in force at the commitment "
                                          f"({', '.join(bases)})")
    parent = entry.get("parent")
    if parent is not None and parent != CHAMPION:
        from ..common.ids import SID_RE  # noqa: PLC0415
        if not isinstance(parent, str) or not SID_RE.match(parent):
            return Rejected(BAD_ENTRY, f"entry.json parent must be a submission id, {CHAMPION!r} or null, "
                                       f"not {parent!r}")
    return entry


def check_content(blobs: dict, bases, comp_rules) -> Rejected | dict:
    """The submission directory's files and entry.json -> entry, or why it cannot enter."""
    from .github import check_files  # noqa: PLC0415
    problem = check_files(blobs, comp_rules)
    if problem:
        return Rejected(BAD_PR, problem)
    return check_entry(blobs.get("entry.json"), bases, comp_rules)
