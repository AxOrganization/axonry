"""Read submission PRs of the competition repo through the GitHub REST API (no checkout needed).

For each open PR: the files it changes must all be inside one directory submissions/<hotkey>/<entry>/; the
content_digest of that directory at the PR head is what must be committed on chain. Everything is
cached per head commit, so an unchanged PR costs nothing but the listing.
"""
import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from ..common.digest import content_digest_of_blobs

API = "https://api.github.com"
DEFAULT_PATTERN = r"^submissions/5[1-9A-HJ-NP-Za-km-z]{47}/[a-z0-9][a-z0-9_.-]{0,63}/"   # competition rules.json


@dataclass
class PRCandidate:
    number: int
    author: str
    hotkey: str | None
    entry: str | None
    directory: str | None
    problem: str | None                       # why this PR cannot enter (None if fine)
    head: str | None = None                   # head commit sha
    digest: str | None = None                 # content_digest of the directory at the head
    title: str = ""                           # the PR title (shown publicly, sanitized by readers)
    author_id: int | None = None              # the author's numeric GitHub id (logins can be renamed and reused)
    opened_at: int | None = None              # when the PR was opened (unix time)


class GitHub:
    def __init__(self, repo: str, token: str | None = None, api: str = API, branch: str = "master"):
        self.repo, self.token, self.api, self.branch = repo, token, api, branch
        self._digest_cache, self._files_cache, self._blobs_cache = {}, {}, {}

    def _get(self, path, raw=False, method="GET", body=None):
        req = urllib.request.Request(self.api + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(), headers={
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "axonry-intake",
            **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
        return body if raw else json.loads(body)

    def _pages(self, path):
        sep = "&" if "?" in path else "?"
        page = 1
        while True:
            items = self._get(f"{path}{sep}per_page=100&page={page}")
            yield from items
            if len(items) < 100:
                return
            page += 1

    def pr(self, number) -> dict:
        """{"number", "state", "head_sha", "mergeable" (True/False/None while GitHub computes), "author", "author_id",
        "merged"}"""
        p = self._get(f"/repos/{self.repo}/pulls/{number}")
        return {"number": number, "state": p["state"], "head_sha": p["head"]["sha"], "mergeable": p.get("mergeable"),
                "author": p["user"]["login"], "author_id": p["user"]["id"], "merged": bool(p.get("merged"))}

    def lock_tag(self, sid: str, commit: str):
        """Freeze the evaluated content: lightweight tag refs/tags/axonry/<sid> -> commit (idempotent)."""
        ref = f"refs/tags/axonry/{sid}"
        try:
            self._get(f"/repos/{self.repo}/git/refs", method="POST", body={"ref": ref, "sha": commit})
        except urllib.error.HTTPError as e:
            if e.code != 422:          # 422: the ref exists already
                raise
            cur = self._get(f"/repos/{self.repo}/git/ref/tags/axonry/{sid}")
            if cur["object"]["sha"] != commit:
                raise RuntimeError(f"lock tag axonry/{sid} exists at {cur['object']['sha']}, not {commit}")
        return ref

    def merge(self, number: int, commit: str, title: str):
        """Merge a champion's locked commit into the main branch: as its PR's merge while the PR head is that commit
        (GitHub then marks the PR merged and closes it); when the head moved on since (405 / 409), the commit alone,
        whatever the PR head is now. The PR only touches its own directory, so this adds exactly it."""
        try:
            self._get(f"/repos/{self.repo}/pulls/{number}/merge", method="PUT",
                      body={"sha": commit, "commit_title": title, "merge_method": "merge"})
            return
        except urllib.error.HTTPError as e:
            if e.code not in (405, 409):
                raise
        # 201 merged, 204 (empty body) already merged; 409 conflict raises
        self._get(f"/repos/{self.repo}/merges", raw=True, method="POST",
                  body={"base": self.branch, "head": commit, "commit_message": title})

    def reviews(self, number: int) -> list:
        return list(self._pages(f"/repos/{self.repo}/pulls/{number}/reviews"))

    def review_state(self, number: int, digest: str, directory: str, reviewers) -> tuple:
        """Verdict of the authorized reviewers on this content: ("approved" | "changes" | None, body). A review
        counts for the commit it was made on (commit_id) and, through that commit's directory digest, for any head
        with the same content; the latest such review by an authorized login wins."""
        latest = None
        for r in self.reviews(number):
            if (r.get("user") or {}).get("login") not in reviewers or r.get("state") not in ("APPROVED",
                                                                                            "CHANGES_REQUESTED"):
                continue
            if not r.get("commit_id") or self.dir_digest(r["commit_id"], directory) != digest:
                continue
            if latest is None or (r.get("submitted_at") or "") >= (latest.get("submitted_at") or ""):
                latest = r
        if latest is None:
            return None, None
        return ("approved" if latest["state"] == "APPROVED" else "changes"), latest.get("body")

    def dismiss_approvals(self, number: int, message: str):
        """Every standing approval of the PR dismissed (the owner withdrew it): its content is reviewed again."""
        for r in self.reviews(number):
            if r.get("state") == "APPROVED":
                self._get(f"/repos/{self.repo}/pulls/{number}/reviews/{r['id']}/dismissals", method="PUT",
                          body={"message": message[:1000], "event": "DISMISS"})

    def post_review(self, number: int, commit: str, event: str, body: str):
        """event: APPROVE | REQUEST_CHANGES. Bound to `commit` (GitHub's commit_id)."""
        return self._get(f"/repos/{self.repo}/pulls/{number}/reviews", method="POST",
                         body={"commit_id": commit, "event": event, "body": body})

    def close(self, number: int):
        return self._get(f"/repos/{self.repo}/pulls/{number}", method="PATCH", body={"state": "closed"})

    def comment(self, number: int, text: str):
        return self._get(f"/repos/{self.repo}/issues/{number}/comments", method="POST", body={"body": text})

    def open_prs(self):
        return list(self._pages(f"/repos/{self.repo}/pulls?state=open"))

    def pr_files(self, number):
        return [f["filename"] for f in self._pages(f"/repos/{self.repo}/pulls/{number}/files")]

    def dir_blobs(self, commit, directory):
        """{relative path: bytes} of every file under `directory` at `commit` (a commit's content never changes:
        cached)."""
        key = (commit, directory)
        if key not in self._blobs_cache:
            self._blobs_cache[key] = self._dir_blobs(commit, directory)
            while len(self._blobs_cache) > 256:          # submissions are small; keep the recent ones
                self._blobs_cache.pop(next(iter(self._blobs_cache)))
        return self._blobs_cache[key]

    def _dir_blobs(self, commit, directory):
        tree = self._get(f"/repos/{self.repo}/git/trees/{commit}?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError(f"tree of {commit} truncated")
        out = {}
        for e in tree["tree"]:
            if e["type"] == "blob" and e["path"].startswith(directory):
                blob = self._get(f"/repos/{self.repo}/git/blobs/{e['sha']}")
                out[e["path"][len(directory):]] = base64.b64decode(blob["content"])
        return out

    def dir_digest(self, commit, directory):
        key = (commit, directory)
        if key not in self._digest_cache:
            self._digest_cache[key] = content_digest_of_blobs(self.dir_blobs(commit, directory))
        return self._digest_cache[key]


def submission_dir(files, path_pattern=DEFAULT_PATTERN):
    """-> (hotkey, entry, directory, problem). All changed files must sit in one directory matching the competition
    rules' submission.path_pattern (submissions/<hotkey>/<entry>/)."""
    pat = re.compile(path_pattern)
    dirs = set()
    for f in files:
        m = pat.match(f)
        if not m:
            return None, None, None, f"changes a path outside submissions/<hotkey>/<entry>/: {f}"
        dirs.add(m.group(0))
    if len(dirs) != 1:
        return None, None, None, f"changes {len(dirs)} submission directories (exactly one allowed)"
    d = dirs.pop()
    _, hotkey, entry, _ = d.split("/")
    return hotkey, entry, d, None


def check_files(blobs: dict, comp_rules: dict) -> str | None:
    """PR content against the competition rules' submission section (required files, types, size)."""
    s = comp_rules["submission"]
    required = s["required"]
    missing = [f for f in required if f not in blobs]
    if missing:
        return f"missing {', '.join(missing)}"
    bad = [r for r in blobs if "." + r.rsplit(".", 1)[-1] not in s["allowed_extensions"]]
    if bad:
        return f"file type not allowed: {', '.join(sorted(bad)[:5])}"
    binary = [r for r, v in blobs.items() if not _text(v)]      # every allowed type is text: a binary is disguised
    if binary:
        return f"not readable text (UTF-8): {', '.join(sorted(binary)[:5])}"
    size = sum(len(v) for v in blobs.values())
    if size > s["max_total_bytes"]:
        return f"{size} bytes > {s['max_total_bytes']}"
    return None


def _text(data: bytes) -> bool:
    try:
        return "\x00" not in data.decode("utf-8")
    except UnicodeDecodeError:
        return False


def _unix(iso: str | None) -> int | None:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()) if iso else None


def scan(gh: GitHub, comp_rules: dict) -> list[PRCandidate]:
    out = []
    for pr in gh.open_prs():
        n, head = pr["number"], pr["head"]["sha"]
        try:
            if (n, head) not in gh._files_cache:
                gh._files_cache[(n, head)] = gh.pr_files(n)
            hk, entry, d, problem = submission_dir(gh._files_cache[(n, head)], comp_rules["submission"]["path_pattern"])
            c = PRCandidate(n, pr["user"]["login"], hk, entry, d, problem, head, title=pr.get("title") or "",
                            author_id=pr["user"]["id"], opened_at=_unix(pr.get("created_at")))
            if not problem:
                c.digest = gh.dir_digest(head, d)
        except (urllib.error.URLError, RuntimeError) as e:
            c = PRCandidate(n, pr["user"]["login"], None, None, None, f"github error: {e}")
        out.append(c)
    return out
