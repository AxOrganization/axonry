import subprocess

import pytest

from axonry.common.canon import sha256_hex
from axonry.orchestrator.cli import competition_rules


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, name, text):
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-qm", name)
    return git(repo, "rev-parse", "HEAD")


def test_later_toolkit_commit_keeps_the_pin(tmp_path):
    git(tmp_path, "init", "-q")
    pinned = commit(tmp_path, "rules.json", "{}")
    pin = {"competition": {"commit": pinned, "rules_sha256": sha256_hex(b"{}")}}
    commit(tmp_path, "gen.py", "x = 1")
    assert competition_rules(tmp_path, pin) == {}
    commit(tmp_path, "rules.json", '{"a": 1}')          # the protocol changed: a new rules version is needed
    with pytest.raises(SystemExit):
        competition_rules(tmp_path, pin)
