"""The automatic reviewer: workspace, comparison set, verdict -> GitHub review."""
import json
import sys
from pathlib import Path

from axonry.common.digest import content_digest_of_blobs
from axonry.registry import Rules, latest_rules
from axonry.review import auto

MINE = {"entry.json": b'{"base": "x"}', "METHOD.md": b"my method"}
OTHER = {"entry.json": b'{"base": "x"}', "METHOD.md": b"an earlier method"}


class GH:
    def __init__(self):
        self.trees = {"c1": OTHER, "c2": MINE, "c0": {"METHOD.md": b"mine before"}}
        self.posted, self.verdict = [], None

    def dir_blobs(self, commit, directory):
        return dict(self.trees[commit])

    def review_state(self, pr, digest, directory, reviewers):
        return self.verdict, None

    def post_review(self, pr, commit, event, body):
        self.posted.append((pr, commit, event, body))
        self.verdict = "approved"


class Reader:
    def __init__(self, doc):
        self.doc = doc

    def get(self, key):
        return self.doc


ITEM = {"pr": 7, "sid": "0000000200-0001", "hotkey": "B", "commit": "c2", "directory": "submissions/B/e/",
        "content_digest": content_digest_of_blobs(MINE), "priority": [200, 1]}
CONTENTS = [{"sid": "0000000100-0001", "hotkey": "A", "pr": 3, "status": "QUEUED", "commit": "c1",
             "directory": "submissions/A/e/", "priority": [100, 1]},
            {"sid": None, "hotkey": "B", "pr": 6, "status": "WAITING", "commit": "c0",       # own earlier version
             "directory": "submissions/B/e/", "priority": [150, 1]},
            {"sid": "0000000300-0001", "hotkey": "C", "pr": 9, "status": "REVIEW", "commit": "c1",  # later
             "directory": "submissions/C/e/", "priority": [300, 1]},
            {**ITEM, "status": "REVIEW"}]


def fake_model(tmp_path, verdict):
    script = tmp_path / "model.py"
    script.write_text("import json, sys, pathlib\nsys.stdin.read()\n"
                      f"pathlib.Path(sys.argv[1]).write_text(json.dumps({verdict!r}))\n")
    return f"{sys.executable} {script} {{out}}"


def rules():
    return latest_rules()


def test_compares_with_earlier_content_of_other_hotkeys_only(tmp_path):
    assert [c["pr"] for c in auto.earlier_than(ITEM, CONTENTS)] == [3]
    ws = auto.prepare(GH(), ITEM, CONTENTS, rules(), tmp_path)
    assert (ws / "submission" / "METHOD.md").read_bytes() == b"my method"
    assert (ws / "earlier" / "100-1-pr3" / "METHOD.md").read_bytes() == b"an earlier method"
    assert "100-1-pr3" in (ws / "earlier" / "INDEX.md").read_text()
    assert "math500" in (ws / "PROMPT.md").read_text()
    assert not (ws / "original").exists()                       # not a reduction


def test_a_reduction_is_reviewed_beside_the_content_first_revealed(tmp_path):
    gh = GH()
    gh.trees["c3"] = {"entry.json": MINE["entry.json"], "METHOD.md": b"my method\nwith a part to delete\n"}
    item = {**ITEM, "revealed": {"commit": "c3", "directory": "submissions/B/e/"}}
    ws = auto.prepare(gh, item, CONTENTS, rules(), tmp_path)
    assert (ws / "submission" / "METHOD.md").read_bytes() == b"my method"
    assert (ws / "original" / "METHOD.md").read_bytes() == b"my method\nwith a part to delete\n"
    assert "`original/`" in (ws / "PROMPT.md").read_text()


def test_approves_only_when_all_checks_pass(tmp_path):
    ok = {k: {"ok": True, "reason": "fine"} for k in ("substantive", "conduct")}
    good = {**ok, "original": {"ok": True, "reason": "new idea", "copies": None}, "message": "Looks good.", "read": True}
    gh = GH()
    auto.run(Reader({"review": [ITEM], "contents": CONTENTS}), gh, rules(), tmp_path, fake_model(tmp_path, good),
             once=True)
    assert gh.posted[0][:3] == (7, "c2", "APPROVE")
    auto.run(Reader({"review": [ITEM], "contents": CONTENTS}), gh, rules(), tmp_path, fake_model(tmp_path, good),
             once=True)
    assert len(gh.posted) == 1                       # already reviewed: never twice

    copy = {**good, "original": {"ok": False, "reason": "same procedure", "copies": "100-1-pr3"}}
    gh = GH()
    auto.run(Reader({"review": [ITEM], "contents": CONTENTS}), gh, rules(), tmp_path / "b", fake_model(tmp_path, copy),
             once=True)
    event, body = gh.posted[0][2:]
    assert event == "REQUEST_CHANGES" and "100-1-pr3" in body and "| Original | **Fail** |" in body

    blind = {**good, "read": False, "substantive": {"ok": False, "reason": "could not read the files"}}
    gh = GH()
    auto.run(Reader({"review": [ITEM], "contents": CONTENTS}), gh, rules(), tmp_path / "c", fake_model(tmp_path, blind),
             once=True)
    assert gh.posted == []                           # a reviewer that could not read posts nothing: retried


def test_a_failing_review_is_left_to_a_human(tmp_path):
    gh = GH()
    for _ in range(auto.MAX_ATTEMPTS + 1):
        auto.run(Reader({"review": [ITEM], "contents": []}), gh, rules(), tmp_path, "false", once=True)
    assert gh.posted == [] and json.loads((tmp_path / "attempts.json").read_text())[
        f"7:{ITEM['content_digest']}"] == auto.MAX_ATTEMPTS
    assert isinstance(rules(), Rules)


def test_relative_workdir(tmp_path, monkeypatch):
    """The model runs with the workspace as its cwd: every path handed to it must be absolute."""
    monkeypatch.chdir(tmp_path)
    ok = {k: {"ok": True, "reason": "fine"} for k in ("substantive", "conduct")}
    good = {**ok, "original": {"ok": True, "reason": "new", "copies": None}, "message": "ok", "read": True}
    gh = GH()
    auto.run(Reader({"review": [ITEM], "contents": []}), gh, rules(), Path("rel/review"), fake_model(tmp_path, good),
             once=True)
    assert gh.posted and gh.posted[0][2] == "APPROVE"


def test_an_unreadable_status_does_not_stop_the_reviewer(tmp_path):
    from axonry.review import auto

    class Down:
        def get(self, key):
            raise ConnectionError("Connection broken: IncompleteRead")
    auto.run(Down(), None, {"review": {"reviewers": ["r"]}}, tmp_path, once=True)    # no exception


def test_the_review_says_whether_deleting_can_fix_it():
    ok = {"ok": True, "reason": "fine"}
    bad = {"ok": False, "reason": "lists benchmark scores"}
    base = {"substantive": ok, "conduct": bad, "original": {**ok, "copies": None}, "message": "Delete lines 90-102."}
    _, body = auto.review_body({**base, "fixable": True}, ITEM)
    assert "Fix by deletion: possible" in body and auto.fixable(body)
    _, body = auto.review_body({**base, "fixable": False}, ITEM)
    assert "not possible" in body and not auto.fixable(body)
    copy = {**base, "conduct": ok, "original": {"ok": False, "reason": "same method", "copies": "100-1-pr3"}}
    _, body = auto.review_body({**copy, "fixable": True}, ITEM)       # a copy is never fixable, whatever the model said
    assert not auto.fixable(body)
    assert auto.fixable("an older review, no mark")


def test_earlier_ideas_are_listed_and_each_review_keeps_its_own(tmp_path):
    ws = auto.prepare(GH(), ITEM, CONTENTS, rules(), tmp_path)
    assert "(no summary: read it in full)" in (ws / "earlier" / "INDEX.md").read_text()
    ideas = {auto.key(CONTENTS[0]): "ablate a refusal direction found from contrast pairs"}
    ws = auto.prepare(GH(), ITEM, CONTENTS, rules(), tmp_path / "b", ideas)
    assert "idea: ablate a refusal direction" in (ws / "earlier" / "INDEX.md").read_text()
    ok = {k: {"ok": True, "reason": "fine"} for k in ("substantive", "conduct")}
    good = {**ok, "original": {"ok": True, "reason": "new", "copies": None}, "message": "ok", "read": True,
            "idea": "scale\nthe   loop heads"}
    auto.run(Reader({"review": [ITEM], "contents": CONTENTS}), GH(), rules(), tmp_path / "c",
             fake_model(tmp_path, good), once=True)
    assert auto.load_ideas(tmp_path / "c")[auto.key(ITEM)] == "scale the loop heads"     # one line


def test_an_idle_reviewer_summarizes_a_content_without_an_idea(tmp_path):
    gh = GH()
    gh.verdict = "approved"                                   # nothing waits for review
    auto.run(Reader({"review": [], "contents": CONTENTS}), gh, rules(), tmp_path,
             fake_model(tmp_path, {"read": True, "idea": "an earlier idea"}), once=True)
    assert auto.load_ideas(tmp_path)[auto.key(CONTENTS[0])] == "an earlier idea"
    for _ in range(auto.MAX_ATTEMPTS + 1):                    # a summary that keeps failing is given up
        auto.run(Reader({"review": [], "contents": CONTENTS}), gh, rules(), tmp_path / "b", "false", once=True)
    assert json.loads((tmp_path / "b" / "attempts.json").read_text())["idea:" + auto.key(CONTENTS[0])] == \
        auto.MAX_ATTEMPTS
