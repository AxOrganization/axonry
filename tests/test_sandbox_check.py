import json

import pytest


def test_axonry_check_refuses_a_parent_under_another_name(tmp_path):
    """Admission accepts parent/edit.json or champion/edit.json only: the check says so before the run ends."""
    from axonry import sandbox_tools
    (tmp_path / "task.json").write_text(json.dumps({"base": "b", "parent": {"sid": "0009538688-0019"}}))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "parent": "0009538688-0019/edit.json", "ops": []}))
    (tmp_path / "0009538688-0019" / "edit").mkdir(parents=True)          # it is there, under another name
    with pytest.raises(SystemExit, match="parent/edit.json"):
        sandbox_tools.main(["check", str(up), "--task", str(tmp_path / "task.json"), "--base-dir", "unused"])


@pytest.mark.parametrize("bad", ["missing", "upload/edit.json"])
def test_axonry_check_says_which_directory_to_give(tmp_path, bad):
    from axonry import sandbox_tools
    (tmp_path / "task.json").write_text(json.dumps({"base": "b"}))
    up = tmp_path / "upload"
    up.mkdir()
    (up / "edit.json").write_text(json.dumps({"base": "b", "ops": []}))
    with pytest.raises(SystemExit, match="edit/ directory"):
        sandbox_tools.main(["check", str(tmp_path / bad), "--task", str(tmp_path / "task.json"), "--base-dir", "x"])


def test_the_parent_is_a_start_of_the_bundle_by_its_id_or_the_base_by_its_full_name():
    from axonry.sandbox_tools import start_of
    task = {"starts": [{"sid": "P1", "role": "parent"}, {"sid": "C1", "role": "champion"}], "base_name": "Org/B@1"}
    assert start_of(task, "C1/edit.json") == "C1" and start_of(task, "Org/B@1") is None
    for bad in (None, "parent/edit.json", "X9/edit.json", "Org/B"):        # never left out, never another
        with pytest.raises(SystemExit, match='"P1/edit.json" or "C1/edit.json" or "Org/B@1"'):
            start_of(task, bad)
    assert start_of({}, None) is None and start_of({}, "champion/edit.json") == "champion"   # a run started before
