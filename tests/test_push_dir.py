"""SshExecutor.push_dir: a build placed on the machine by content - what it holds already is not sent again."""
import subprocess
from pathlib import Path

from axonry.evaluation import executor as ex_mod
from axonry.evaluation.executor import SshExecutor


def machine_in(tmp_path, monkeypatch):
    sent = []
    ex = SshExecutor(lambda on: object())
    ex.root = str(tmp_path / "remote")
    ex._ssh = lambda on, cmd, timeout=600, binary=False: subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout

    def scp(m, key, srcs, dst, timeout=1800, attempts=3):
        for s in srcs:
            sent.append(Path(s).name)
            Path(dst).write_bytes(Path(s).read_bytes())
    monkeypatch.setattr(ex_mod.machines, "scp", scp)
    return ex, sent


def build(tmp_path, name, own):
    d = tmp_path / name
    (d / "S1" / "edit").mkdir(parents=True)
    (d / "edit.json").write_text(own)
    (d / "S1" / "edit" / "edit.npz").write_bytes(b"the champion's archive" * 1000)
    return d


def test_a_start_placed_in_many_builds_is_sent_once(tmp_path, monkeypatch):
    ex, sent = machine_in(tmp_path, monkeypatch)
    ex.push_dir("eval", build(tmp_path, "a", "one"), "0000000001-0001", "edit")
    ex.push_dir("eval", build(tmp_path, "b", "two"), "0000000002-0001", "edit")
    assert sorted(sent) == ["edit.json", "edit.json", "edit.npz"]  # the shared start once, each own spec
    for sid, own in (("0000000001-0001", "one"), ("0000000002-0001", "two")):
        d = Path(ex.root) / "eval" / sid / "edit"
        assert (d / "edit.json").read_text() == own and (d / "S1" / "edit" / "edit.npz").exists()


def test_a_cut_transfer_keeps_what_completed_and_a_retry_sends_the_rest(tmp_path, monkeypatch):
    ex, sent = machine_in(tmp_path, monkeypatch)
    plain = ex_mod.machines.scp
    done = []

    def cut(m, key, srcs, dst, timeout=1800, attempts=3):
        if done:                                                   # the second file sent: the link drops
            raise ex_mod.machines.MachineError("eval: scp timed out")
        plain(m, key, srcs, dst)
        done.append(Path(srcs[0]).name)
    monkeypatch.setattr(ex_mod.machines, "scp", cut)
    d = build(tmp_path, "a", "one")
    try:
        ex.push_dir("eval", d, "0000000001-0001", "edit")
        raise AssertionError("expected the cut")
    except ex_mod.ExecError:
        pass
    monkeypatch.setattr(ex_mod.machines, "scp", plain)
    sent.clear()
    ex.push_dir("eval", d, "0000000001-0001", "edit")
    assert len(done) == 1 and sent == [n for n in ("edit.json", "edit.npz") if n not in done]   # only the rest
    out = Path(ex.root) / "eval" / "0000000001-0001" / "edit"
    assert (out / "edit.json").read_text() == "one" and (out / "S1" / "edit" / "edit.npz").exists()

def test_the_same_content_at_two_paths_is_sent_once_and_placed_at_both(tmp_path, monkeypatch):
    ex, sent = machine_in(tmp_path, monkeypatch)
    d = build(tmp_path, "a", "one")
    (d / "S1" / "edit" / "REPORT.md").write_text("one")             # the same bytes as edit.json
    ex.push_dir("eval", d, "0000000001-0001", "edit")
    out = Path(ex.root) / "eval" / "0000000001-0001" / "edit"
    assert (out / "edit.json").read_text() == (out / "S1" / "edit" / "REPORT.md").read_text() == "one"
    assert sorted(sent) == ["REPORT.md", "edit.npz"] or sorted(sent) == ["edit.json", "edit.npz"]
